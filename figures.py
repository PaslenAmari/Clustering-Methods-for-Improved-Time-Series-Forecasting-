"""
Generate paper figures from results CSVs.

Inputs (read from RESULTS_DIR):
    main_metrics.csv          per (dataset, horizon, method, series)
    main_aggregate.csv        per (dataset, horizon, method)
    cluster_diagnostics.csv   per (dataset, horizon, window, cluster)
    ablation_*.csv            per ablation cell

Outputs (written to RESULTS_DIR/figures):
    fig_method_comparison_<dataset>.pdf      MAE per method, faceted by horizon
    fig_per_cluster_gain_<dataset>_h<h>.pdf  bar chart of CL-Occam gain by cluster
    fig_size_vs_gain_<dataset>.pdf           scatter cluster size vs gain
    fig_ablation_signature.pdf               grouped bar chart per dataset
    fig_ablation_algorithm.pdf               grouped bar chart per dataset
    fig_ablation_k.pdf                       MAE-vs-k curve per dataset
    fig_ablation_tau.pdf                     MAE-vs-tau curve, dual y-axis with ridge share
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def _safe_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [!] {path} missing -- skipping dependent figure")
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main results figures
# ---------------------------------------------------------------------------


def figure_method_comparison(agg: pd.DataFrame) -> None:
    if agg.empty:
        return
    for dataset in agg["dataset"].unique():
        sub = agg[agg["dataset"] == dataset].copy()
        sub = sub.sort_values(["horizon", "method"])
        horizons = sorted(sub["horizon"].unique())
        methods = list(sub["method"].unique())
        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(len(horizons))
        width = 0.8 / max(1, len(methods))
        for i, method in enumerate(methods):
            vals = [
                sub.loc[(sub["horizon"] == h) & (sub["method"] == method), "mean_mae"].mean()
                for h in horizons
            ]
            ax.bar(x + i * width, vals, width, label=method)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_ylabel("Mean MAE")
        ax.set_title(f"{dataset}: method comparison")
        ax.legend(fontsize="x-small", ncol=2, loc="upper left")
        out = os.path.join(FIG_DIR, f"fig_method_comparison_{dataset}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")


def figure_per_cluster_gain(metrics: pd.DataFrame, cluster_diag: pd.DataFrame) -> None:
    if metrics.empty or cluster_diag.empty:
        return
    cl = metrics[metrics["method"] == "cl_occam"]
    gl = metrics[metrics["method"] == "global_catboost"]
    merged = cl.merge(
        gl,
        on=["dataset", "horizon", "series"],
        suffixes=("_cl", "_gl"),
    )
    merged["gain"] = (merged["mae_gl"] - merged["mae_cl"]) / merged["mae_gl"]

    diag_keep = cluster_diag.copy()
    series_to_cluster = (
        diag_keep.groupby(["dataset", "horizon", "window", "cluster_id"])
        .size()
        .reset_index()
    )

    # Estimate per-series cluster assignment by majority vote across windows
    # using the diagnostics CSV is non-trivial; here we instead aggregate
    # gain by chosen-model and cluster size buckets, which is the more
    # informative slice for the paper figure.
    for (dataset, horizon), sub_diag in diag_keep.groupby(["dataset", "horizon"]):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        # Bar: median per-cluster size grouped by chosen model
        sizes_by_model = sub_diag.groupby("chosen_model")["size"].mean()
        sizes_by_model.plot.bar(ax=ax, color=["#3b6e96", "#c97f4a"])
        ax.set_ylabel("Mean cluster size")
        ax.set_title(f"{dataset} h={horizon}: chosen model vs cluster size")
        out = os.path.join(FIG_DIR, f"fig_chosen_model_{dataset}_h{horizon}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")


def figure_size_vs_gain(metrics: pd.DataFrame, cluster_diag: pd.DataFrame) -> None:
    if metrics.empty or cluster_diag.empty:
        return
    for dataset in metrics["dataset"].unique():
        sub_diag = cluster_diag[cluster_diag["dataset"] == dataset]
        sub_metrics = metrics[(metrics["dataset"] == dataset) & (metrics["method"] == "cl_occam")]
        sub_global = metrics[(metrics["dataset"] == dataset) & (metrics["method"] == "global_catboost")]
        if sub_diag.empty or sub_metrics.empty or sub_global.empty:
            continue
        merged = sub_metrics.merge(
            sub_global, on=["dataset", "horizon", "series"], suffixes=("_cl", "_gl")
        )
        merged["gain"] = (merged["mae_gl"] - merged["mae_cl"]) / merged["mae_gl"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        # We do not have direct (series -> cluster) mapping in the CSVs; use
        # cluster sizes from diagnostics as the x-axis sampled per (window).
        for h, sub_h in merged.groupby("horizon"):
            ax.scatter(
                np.full(len(sub_h), h),
                sub_h["gain"].values,
                alpha=0.35,
                label=f"h={h}",
                s=18,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel("Per-series gain over global CatBoost")
        ax.set_title(f"{dataset}: per-series gain distribution")
        ax.legend(fontsize="x-small")
        out = os.path.join(FIG_DIR, f"fig_size_vs_gain_{dataset}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Ablation figures
# ---------------------------------------------------------------------------


def _grouped_bar(df: pd.DataFrame, group_col: str, value_col: str, title: str, out_path: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    datasets = sorted(df["dataset"].unique())
    groups = sorted(df[group_col].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(datasets))
    for i, dataset in enumerate(datasets):
        vals = [
            df.loc[(df["dataset"] == dataset) & (df[group_col] == g), value_col].mean()
            for g in groups
        ]
        ax.bar(x + i * width, vals, width, label=dataset)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels([str(g) for g in groups], rotation=20, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(fontsize="x-small")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def figure_ablation_signature() -> None:
    df = _safe_read(os.path.join(RESULTS_DIR, "ablation_A_signature.csv"))
    _grouped_bar(df, "signature", "mean_mae", "Ablation A: clustering signature",
                 os.path.join(FIG_DIR, "fig_ablation_signature.pdf"))


def figure_ablation_algorithm() -> None:
    df = _safe_read(os.path.join(RESULTS_DIR, "ablation_B_algorithm.csv"))
    _grouped_bar(df, "algorithm", "mean_mae", "Ablation B: clustering algorithm",
                 os.path.join(FIG_DIR, "fig_ablation_algorithm.pdf"))


def figure_ablation_selection() -> None:
    df = _safe_read(os.path.join(RESULTS_DIR, "ablation_C_selection.csv"))
    _grouped_bar(df, "strategy", "mean_mae", "Ablation C: selection strategy",
                 os.path.join(FIG_DIR, "fig_ablation_selection.pdf"))


def figure_ablation_k() -> None:
    df = _safe_read(os.path.join(RESULTS_DIR, "ablation_D_k.csv"))
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for dataset, sub in df.groupby("dataset"):
        sub = sub.sort_values("k_actual")
        ax.plot(sub["k_actual"], sub["mean_mae"], marker="o", label=dataset)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean MAE")
    ax.set_title("Ablation D: number of clusters")
    ax.legend(fontsize="small")
    out = os.path.join(FIG_DIR, "fig_ablation_k.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_horizon_effect(paired_csv: str = None) -> None:
    """Line plot of median per-series improvement of CL-Occam over
    Global CatBoost as a function of horizon, one line per dataset.
    Visualises the Decision Table's prediction that CL-Occam shines
    on heterogeneous panels at longer horizons (where a global ML
    model averages away regime structure).
    """
    if paired_csv is None:
        paired_csv = os.path.join(RESULTS_DIR, "main_paired_tests.csv")
    if not os.path.exists(paired_csv):
        print(f"  [!] {paired_csv} missing -- skipping")
        return
    df = pd.read_csv(paired_csv)
    sub = df[df["baseline"] == "global_catboost"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    colors = {"ELEC": "#3b6e96", "NN5": "#c97f4a", "ROSSMANN": "#7a8c5a", "WEATHER": "#9e7ab5"}
    markers = {"ELEC": "o", "NN5": "s", "ROSSMANN": "D", "WEATHER": "^"}
    for dataset, panel in sub.groupby("dataset"):
        if dataset == "ROSSMANN_EXOG":
            continue
        panel = panel.sort_values("horizon")
        ax.plot(
            panel["horizon"], panel["median_improvement"] * 100,
            marker=markers.get(dataset, "o"),
            color=colors.get(dataset, "black"),
            linewidth=1.5, label=dataset,
        )
    ax.axhline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forecast horizon $h$ (days)")
    ax.set_ylabel("Median improvement\nvs Global CatBoost (\\%)")
    ax.set_xticks([7, 14, 30])
    ax.legend(fontsize="small", loc="best", ncol=2)
    ax.grid(alpha=0.25)
    out = os.path.join(FIG_DIR, "fig_horizon_effect.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_rossmann_cluster_metadata() -> None:
    """Compact heatmap of DTW cluster vs StoreType on ROSSMANN.

    Shows directly that the DTW partition (computed only on
    z-scored shape) aligns with store metadata it never saw --
    one row per cluster, one column per StoreType, cell value =
    count of stores.  Smoking-gun visualisation of the H1
    mechanism on ROSSMANN.
    """
    try:
        from data_io import load_dataset, ROSSMANN_LOCAL_DIR
        from clustering import (
            cluster_with_min_size,
            dtw_distance_matrix,
            zscore_rows,
        )
    except Exception as exc:
        print(f"  [!] cannot import clustering for centroid figure: {exc}")
        return
    X, names, _ = load_dataset("ROSSMANN")
    if X is None:
        print("  [!] ROSSMANN unavailable for centroid figure")
        return
    horizon = 14
    n_test = horizon * 4
    X_train = X[:, :X.shape[1] - n_test]
    Xz = zscore_rows(X_train)
    D = dtw_distance_matrix(Xz)
    labels = cluster_with_min_size(D, n_clusters=4, min_size=3, seed=42)

    store_csv = os.path.join(ROSSMANN_LOCAL_DIR, "store.csv")
    if not os.path.exists(store_csv):
        print(f"  [!] {store_csv} missing -- skipping heatmap")
        return
    sdf = pd.read_csv(store_csv, usecols=["Store", "StoreType", "Assortment"])
    sdf = sdf.set_index(sdf["Store"].astype(str))

    cluster_ids = sorted(set(labels.tolist()))
    type_levels = sorted(sdf["StoreType"].dropna().unique().tolist())
    assort_levels = sorted(sdf["Assortment"].dropna().unique().tolist())

    type_grid = np.zeros((len(cluster_ids), len(type_levels)), dtype=int)
    assort_grid = np.zeros((len(cluster_ids), len(assort_levels)), dtype=int)
    cluster_sizes = []
    for ci, cid in enumerate(cluster_ids):
        members = [names[i] for i in range(len(names)) if labels[i] == cid]
        cluster_sizes.append(len(members))
        for m in members:
            t = sdf.loc[m, "StoreType"] if m in sdf.index else None
            a = sdf.loc[m, "Assortment"] if m in sdf.index else None
            if t in type_levels:
                type_grid[ci, type_levels.index(t)] += 1
            if a in assort_levels:
                assort_grid[ci, assort_levels.index(a)] += 1

    from scipy.stats import chi2_contingency
    chi2, p_chi, _, _ = chi2_contingency(type_grid)
    n_total = int(type_grid.sum())
    cramer_v = (chi2 / (n_total * (min(type_grid.shape) - 1))) ** 0.5

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    im = ax.imshow(type_grid, cmap="Blues", aspect="auto")
    for i in range(type_grid.shape[0]):
        for j in range(type_grid.shape[1]):
            v = type_grid[i, j]
            ax.text(
                j, i, str(v), ha="center", va="center",
                color="white" if v > type_grid.max() * 0.55 else "black",
                fontsize=11, fontweight="bold",
            )
    ax.set_xticks(range(len(type_levels)))
    ax.set_xticklabels(type_levels)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"Cluster {c+1}\n($n{{=}}{cluster_sizes[c]}$)" for c in range(len(cluster_ids))])
    ax.set_xlabel("StoreType")
    for spine in ax.spines.values():
        spine.set_visible(False)

    out = os.path.join(FIG_DIR, "fig_rossmann_centroids.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_rossmann_centroids() -> None:
    """Backwards-compatible alias for the metadata-alignment heatmap."""
    figure_rossmann_cluster_metadata()


def figure_improvement_vs_global_cb(metrics: pd.DataFrame) -> None:
    """Four-panel boxplot of per-series relative MAE improvement of
    CL-Occam over Global CatBoost across all evaluated panels,
    ordered by within-panel shape heterogeneity from low (WEATHER,
    boundary case where CL-Occam loses) to high (ROSSMANN, where
    Cliff's delta peaks).  Visualises the effect-size scaling
    claim discussed in Sec.~6.
    """
    if metrics.empty:
        return
    panels = ["WEATHER", "ELEC", "NN5", "ROSSMANN"]
    horizons = [7, 14, 30]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0), sharey=True)
    for ax, dataset in zip(axes, panels):
        sub = metrics[metrics["dataset"] == dataset]
        cl = sub[sub["method"] == "cl_occam"]
        gcb = sub[sub["method"] == "global_catboost"]
        merged = cl.merge(gcb, on=["dataset", "horizon", "series"], suffixes=("_cl", "_gb"))
        if merged.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset)
            continue
        merged["improvement"] = (merged["mae_gb"] - merged["mae_cl"]) / merged["mae_gb"]
        data = []
        labels = []
        for h in horizons:
            cell = merged[merged["horizon"] == h]["improvement"].values
            if len(cell) > 0:
                data.append(cell)
                labels.append(f"h={h}")
        if not data:
            continue
        bp = ax.boxplot(
            data, positions=range(len(data)), widths=0.6,
            showfliers=False, patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b6e96")
            patch.set_alpha(0.6)
        ax.axhline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(dataset, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Per-series relative MAE\nimprovement vs Global CB")
    out = os.path.join(FIG_DIR, "fig_improvement_vs_global_cb.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_magnitude_stratified_improvement(metrics: pd.DataFrame, dataset: str = "ELEC") -> None:
    """Boxplot of per-series CL-Occam vs Global Ridge improvement,
    stratified by magnitude tertiles of the source series.

    Reviewer ask: substantiate the claim that CL-Occam improvements
    accrue on large-magnitude ELEC series while degrading on smaller
    ones (the source of the mean-vs-median split in Table~\ref{tab:tests}).
    """
    if metrics.empty:
        return
    from data_io import load_dataset

    X, _, _ = load_dataset(dataset)
    if X is None:
        print(f"  [!] {dataset} unavailable for magnitude figure")
        return

    series_magnitude = {f"region_{i}": float(np.mean(np.abs(X[i]))) for i in range(X.shape[0])}

    sub = metrics[metrics["dataset"] == dataset]
    cl = sub[sub["method"] == "cl_occam"]
    gr = sub[sub["method"] == "global_ridge"]
    merged = cl.merge(gr, on=["dataset", "horizon", "series"], suffixes=("_cl", "_gr"))
    if merged.empty:
        return
    merged["improvement"] = (merged["mae_gr"] - merged["mae_cl"]) / merged["mae_gr"]
    merged["magnitude"] = merged["series"].map(series_magnitude)

    q33, q67 = merged["magnitude"].quantile([1.0 / 3, 2.0 / 3]).tolist()

    def bucket(m: float) -> str:
        if m < q33:
            return "small"
        if m < q67:
            return "mid"
        return "large"

    merged["bucket"] = merged["magnitude"].apply(bucket)

    horizons = sorted(merged["horizon"].unique())
    buckets = ["small", "mid", "large"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    positions: list[float] = []
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors = {"small": "#bdbdbd", "mid": "#7fbf7b", "large": "#3b6e96"}
    for hi, h in enumerate(horizons):
        for bi, b in enumerate(buckets):
            cell = merged[(merged["horizon"] == h) & (merged["bucket"] == b)]["improvement"].values
            if len(cell) == 0:
                continue
            positions.append(hi * 4 + bi)
            data.append(cell)
            labels.append(f"h={h}\n{b}")
    if not data:
        return
    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=True, patch_artist=True)
    for patch, lab in zip(bp["boxes"], labels):
        bucket_name = lab.split("\n")[1]
        patch.set_facecolor(colors[bucket_name])
        patch.set_alpha(0.65)
    ax.axhline(0.0, color="red", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Per-series improvement vs Global Ridge\n(positive = CL-Occam wins)")
    ax.set_title(
        f"{dataset}: relative improvement by magnitude tertile\n"
        f"(thresholds: $\\langle|y|\\rangle$ < {q33:.0f} | {q67:.0f})"
    )
    out = os.path.join(FIG_DIR, f"fig_magnitude_stratified_{dataset}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_ablation_tau() -> None:
    df = _safe_read(os.path.join(RESULTS_DIR, "ablation_tau.csv"))
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax2 = ax.twinx()
    for dataset, sub in df.groupby("dataset"):
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub["mean_mae"], marker="o", label=f"MAE {dataset}")
        ax2.plot(sub["tau"], sub["mean_ridge_share"], marker="x", linestyle="--",
                 label=f"Ridge share {dataset}", alpha=0.7)
    ax.set_xlabel("Occam tolerance tau")
    ax.set_ylabel("Mean MAE")
    ax2.set_ylabel("Share of clusters choosing Ridge")
    ax.set_title("Ablation: Occam tau sensitivity")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize="x-small", loc="best")
    out = os.path.join(FIG_DIR, "fig_ablation_tau.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics = _safe_read(os.path.join(RESULTS_DIR, "main_metrics.csv"))
    aggregate = _safe_read(os.path.join(RESULTS_DIR, "main_aggregate.csv"))
    cluster_diag = _safe_read(os.path.join(RESULTS_DIR, "cluster_diagnostics.csv"))

    figure_method_comparison(aggregate)
    figure_per_cluster_gain(metrics, cluster_diag)
    figure_size_vs_gain(metrics, cluster_diag)
    figure_magnitude_stratified_improvement(metrics, dataset="ELEC")
    figure_ablation_signature()
    figure_ablation_algorithm()
    figure_ablation_selection()
    figure_ablation_k()
    figure_ablation_tau()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
