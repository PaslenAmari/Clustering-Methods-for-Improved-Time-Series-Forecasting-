"""
Per-cluster diagnostics + computational cost figures (paper section 7
and Discussion).

Reads:
    results/seed_NN5.json   from seed_runs.py

Writes:
    results/diagnostics_NN5.csv               per (horizon, window, cluster) MAE
    results/figures/fig_per_cluster_mae.pdf   bar chart, faceted by horizon
    results/figures/fig_centroids.pdf         z-scored mean per cluster
    results/figures/fig_silhouette.pdf        silhouette score curve over k
    results/figures/fig_dtw_scaling.pdf       wall-clock vs N (log-log)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

warnings.filterwarnings("ignore")

from clustering import cluster_with_min_size, dtw_distance_matrix, zscore_rows  # noqa: E402
from data_io import load_dataset  # noqa: E402


RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SEED_PATH = os.path.join(RESULTS_DIR, "seed_NN5.json")

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def _series_id(idx: int) -> str:
    return f"region_{idx}"


# ---------------------------------------------------------------------------
# Per-cluster MAE breakdown
# ---------------------------------------------------------------------------


def per_cluster_mae(seed: dict) -> pd.DataFrame:
    rows = []
    for h_str, h_block in seed["horizons"].items():
        horizon = int(h_str)
        for w in h_block["windows"]:
            labels = np.array(w["labels"])
            cl_mae = w["cl_per_series_mae"]
            ridge_mae = w["ridge_per_series_mae"]
            for cid in np.unique(labels):
                members = np.where(labels == cid)[0]
                segs = [_series_id(i) for i in members]
                cl_vals = [cl_mae[s] for s in segs if s in cl_mae]
                ridge_vals = [ridge_mae[s] for s in segs if s in ridge_mae]
                rows.append(
                    {
                        "horizon": horizon,
                        "window": w["window_idx"],
                        "cluster_id": int(cid),
                        "size": int(len(members)),
                        "chosen_model": w["model_mapping"][str(int(cid))],
                        "cl_mean_mae": float(np.mean(cl_vals)) if cl_vals else float("nan"),
                        "ridge_mean_mae": float(np.mean(ridge_vals)) if ridge_vals else float("nan"),
                        "improvement_pct": (
                            100.0 * (np.mean(ridge_vals) - np.mean(cl_vals)) / np.mean(ridge_vals)
                            if cl_vals and ridge_vals
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def figure_per_cluster_mae(df: pd.DataFrame) -> None:
    horizons = sorted(df["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 3.5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        sub = df[df["horizon"] == h]
        # Average across windows for each cluster id (cluster ids are
        # window-stable here; avg is informative).
        agg = sub.groupby("cluster_id").agg(
            size=("size", "mean"),
            cl=("cl_mean_mae", "mean"),
            ridge=("ridge_mean_mae", "mean"),
            chosen=("chosen_model", lambda s: s.mode().iloc[0]),
        ).reset_index()
        x = np.arange(len(agg))
        width = 0.4
        ax.bar(x - width / 2, agg["ridge"], width, label="Global Ridge", color="#a0a0a0")
        ax.bar(x + width / 2, agg["cl"], width, label="CL-Occam", color="#3b6e96")
        for xi, (_, r) in zip(x, agg.iterrows()):
            ax.text(xi, max(r["ridge"], r["cl"]) * 1.02,
                    f"n={int(r['size'])}\n{r['chosen']}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(c)}" for c in agg["cluster_id"]])
        ax.set_title(f"NN5 h={h}")
        ax.set_ylabel("Mean MAE" if ax is axes[0] else "")
    axes[0].legend(fontsize="x-small", loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_per_cluster_mae.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Cluster centroids
# ---------------------------------------------------------------------------


def figure_centroids(seed: dict, X: np.ndarray) -> None:
    fig, axes = plt.subplots(len(seed["horizons"]), 1, figsize=(8, 2.5 * len(seed["horizons"])))
    if len(seed["horizons"]) == 1:
        axes = [axes]
    for ax, (h_str, h_block) in zip(axes, seed["horizons"].items()):
        # Use the first window's training data to plot centroids
        w0 = h_block["windows"][0]
        labels = np.array(w0["labels"])
        X_train = X[:, : w0["test_start"]]
        Xz = (X_train - X_train.mean(axis=1, keepdims=True)) / np.where(
            X_train.std(axis=1, keepdims=True) > 1e-12, X_train.std(axis=1, keepdims=True), 1.0
        )
        for cid in np.unique(labels):
            mask = labels == cid
            centroid = Xz[mask].mean(axis=0)
            # Show last 4 weeks of training to keep the panel readable
            last = centroid[-28:]
            ax.plot(last, label=f"c{int(cid)} (n={mask.sum()})", linewidth=1.0)
        ax.set_title(f"NN5 h={h_str}: cluster centroids (last 28 train days, z-scored)")
        ax.legend(fontsize="x-small", ncol=4, loc="upper right")
        ax.set_xlabel("Day index (within last 4 weeks)")
        ax.set_ylabel("z-score")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_centroids.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Silhouette + within/between variance
# ---------------------------------------------------------------------------


def figure_silhouette(X: np.ndarray) -> pd.DataFrame:
    print("  computing DTW for silhouette curve...")
    D = dtw_distance_matrix(zscore_rows(X))
    rows = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = list(range(2, 9))
    silhouettes = []
    var_ratios = []
    for k in ks:
        labels = cluster_with_min_size(D, n_clusters=k, min_size=3, seed=42)
        if len(np.unique(labels)) < 2:
            silhouettes.append(float("nan"))
            var_ratios.append(float("nan"))
            continue
        s = float(silhouette_score(D, labels, metric="precomputed"))
        # Within/between variance ratio on z-scored series flattened
        Xz = zscore_rows(X)
        means = np.array([Xz[labels == c].mean(axis=0) for c in np.unique(labels)])
        within = float(np.mean([np.mean((Xz[labels == c] - means[i]) ** 2) for i, c in enumerate(np.unique(labels))]))
        between = float(np.mean(np.var(means, axis=0)))
        ratio = within / (within + between) if (within + between) > 0 else float("nan")
        silhouettes.append(s)
        var_ratios.append(ratio)
        rows.append({"k": k, "k_actual": int(labels.max() + 1), "silhouette": s, "var_within_share": ratio})
    ax.plot(ks, silhouettes, "o-", label="Silhouette (DTW)", color="#3b6e96")
    ax2 = ax.twinx()
    ax2.plot(ks, var_ratios, "x--", label="Within-cluster variance share", color="#c97f4a", alpha=0.8)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score (higher = tighter)")
    ax2.set_ylabel("Within / (within + between) variance")
    lines, labels_ = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_ + labels2, fontsize="x-small", loc="best")
    ax.set_title("NN5: cluster quality vs k")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_silhouette.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DTW computational cost
# ---------------------------------------------------------------------------


def figure_dtw_scaling(X: np.ndarray) -> pd.DataFrame:
    print("  measuring DTW scaling on subsampled panels...")
    Ns = [30, 60, 90, 111]
    times = []
    for N in Ns:
        sub = X[:N]
        t = time.time()
        _ = dtw_distance_matrix(zscore_rows(sub))
        elapsed = time.time() - t
        times.append(elapsed)
        print(f"    N={N}: {elapsed:.2f}s")
    Ns_arr = np.array(Ns, dtype=float)
    times_arr = np.array(times, dtype=float)
    # Fit log-log slope: time = a * N^b → log time = log a + b log N
    coeffs = np.polyfit(np.log(Ns_arr), np.log(times_arr), 1)
    slope = float(coeffs[0])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(Ns_arr, times_arr, "o-", color="#3b6e96", label=f"observed (slope {slope:.2f})")
    ax.loglog(Ns_arr, np.exp(coeffs[1]) * Ns_arr ** slope, "--", color="#a0a0a0",
              label=f"fit ~ N^{slope:.2f}")
    ax.set_xlabel("Panel size N")
    ax.set_ylabel("DTW distance matrix wall-clock (s)")
    ax.set_title("DTW scaling on NN5")
    ax.legend(fontsize="small")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_dtw_scaling.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}, slope = {slope:.2f} (theoretical 2.0)")
    return pd.DataFrame({"N": Ns, "time_s": times, "fitted_slope": [slope] * len(Ns)})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    if not os.path.exists(SEED_PATH):
        print(f"  [!] {SEED_PATH} not found; run seed_runs.py first")
        return 1
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(SEED_PATH, "r", encoding="utf-8") as fh:
        seed = json.load(fh)
    X, _, _ = load_dataset(seed["dataset"])

    print("Per-cluster MAE breakdown ...")
    df_pc = per_cluster_mae(seed)
    df_pc.to_csv(os.path.join(RESULTS_DIR, "diagnostics_NN5.csv"), index=False)
    figure_per_cluster_mae(df_pc)

    print("Centroids ...")
    figure_centroids(seed, X)

    print("Silhouette + variance ratio ...")
    df_sil = figure_silhouette(X)
    df_sil.to_csv(os.path.join(RESULTS_DIR, "diagnostics_silhouette.csv"), index=False)

    print("DTW scaling ...")
    df_scale = figure_dtw_scaling(X)
    df_scale.to_csv(os.path.join(RESULTS_DIR, "diagnostics_scaling.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
