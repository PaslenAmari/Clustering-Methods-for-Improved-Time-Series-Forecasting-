"""
Ablation studies for the DAMDID 2026 paper.

Each ablation isolates one design decision while keeping the rest of
the pipeline frozen, and writes a CSV that becomes one table in the
paper.  All ablations run on a single horizon ``ABLATION_HORIZON`` for
budget reasons; the main results table covers the full horizon sweep.

Ablations:
    A. Clustering signature       results/ablation_A_signature.csv
    B. Clustering algorithm       results/ablation_B_algorithm.csv
    C. Selection strategy         results/ablation_C_selection.csv
    D. Number of clusters         results/ablation_D_k.csv
    Occam-tau sensitivity         results/ablation_tau.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans

from clustering import (
    cluster_with_min_size,
    dtw_distance_matrix,
    zscore_rows,
)
from data_io import load_dataset
from eval import mae as mae_metric, mase as mase_metric, smape as smape_metric
from forecast import (
    OCCAM_TAU_LARGE,
    OCCAM_TAU_SMALL,
    forecast_clustered_local,
    select_model_for_cluster,
)
from run_damdid import (
    N_WINDOWS,
    _evaluate_window_forecasts,
    _series_id,
    make_test_windows,
)
from signatures import SIGNATURE_REGISTRY


RESULTS_DIR = "results"
ABLATION_HORIZON = 14


# ---------------------------------------------------------------------------
# Distance computation for non-DTW algorithms
# ---------------------------------------------------------------------------


def euclidean_distance_matrix(S: np.ndarray) -> np.ndarray:
    return squareform(pdist(S, metric="euclidean"))


def cluster_kmedoids(D: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """KMedoids on a precomputed distance matrix.  Lightweight PAM
    variant -- avoids adding sklearn-extra as a dependency."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    for _ in range(50):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            sub = D[np.ix_(members, members)]
            new_medoids[c] = members[np.argmin(sub.sum(axis=1))]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return labels


def cluster_kshape(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """k-Shape clustering via tslearn, falling back to Euclidean+KMeans
    on the z-scored series if tslearn is missing."""
    try:
        from tslearn.clustering import KShape

        Xz = zscore_rows(X).reshape(X.shape[0], X.shape[1], 1)
        ks = KShape(n_clusters=k, random_state=seed, n_init=3)
        return ks.fit_predict(Xz)
    except Exception:
        Xz = zscore_rows(X)
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xz)


# ---------------------------------------------------------------------------
# Backtest evaluation given a partition
# ---------------------------------------------------------------------------


def evaluate_partition(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int,
    labels_per_window: List[np.ndarray],
    mapping_per_window: List[Dict[int, str]],
) -> Dict[str, float]:
    """Run cl-style backtest with provided partitions and model mappings."""
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    window_records: List[Dict[str, Dict[str, float]]] = []
    for window_idx, (test_start, test_end) in enumerate(test_windows):
        X_train = X[:, :test_start]
        labels = labels_per_window[window_idx]
        mapping = mapping_per_window[window_idx]
        try:
            forecasts = forecast_clustered_local(X_train, labels, mapping, start_date, horizon)
        except Exception as exc:
            print(f"  [!] cl forecast failed: {exc}")
            forecasts = {}
        window_records.append(
            _evaluate_window_forecasts(forecasts, X, test_start, test_end, X_train)
        )
    series_metric_lists: Dict[str, Dict[str, List[float]]] = {}
    for window in window_records:
        for seg, metrics in window.items():
            store = series_metric_lists.setdefault(seg, {"mae": [], "smape": [], "mase": []})
            for k in store:
                v = metrics.get(k)
                if v is not None and np.isfinite(v):
                    store[k].append(float(v))
    if not series_metric_lists:
        return {"n_series": 0, "mean_mae": float("nan"), "mean_mase": float("nan"), "mean_smape": float("nan")}
    mae_arr = np.array([np.mean(d["mae"]) for d in series_metric_lists.values() if d["mae"]])
    smape_arr = np.array([np.mean(d["smape"]) for d in series_metric_lists.values() if d["smape"]])
    mase_arr = np.array([np.mean(d["mase"]) for d in series_metric_lists.values() if d["mase"]])
    return {
        "n_series": len(series_metric_lists),
        "mean_mae": float(np.nanmean(mae_arr)),
        "mean_mase": float(np.nanmean(mase_arr)),
        "mean_smape": float(np.nanmean(smape_arr)),
    }


# ---------------------------------------------------------------------------
# Clustering builders for ablations
# ---------------------------------------------------------------------------


def _occam_select_for_cluster(
    X_train: np.ndarray,
    cluster_mask: np.ndarray,
    horizon: int,
    start_date: str,
    candidate_pool: Sequence[str],
) -> str:
    chosen, _ = select_model_for_cluster(
        X_train_full=X_train,
        cluster_mask=cluster_mask,
        horizon=horizon,
        start_date=start_date,
        candidate_pool=candidate_pool,
    )
    return chosen


def build_partition(
    X_train: np.ndarray,
    signature_name: str,
    algorithm: str,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute cluster labels using the requested signature x algorithm pair."""
    if signature_name not in SIGNATURE_REGISTRY:
        raise ValueError(f"Unknown signature {signature_name}")
    S = SIGNATURE_REGISTRY[signature_name](X_train)

    if algorithm == "dtw_agglomerative":
        if signature_name != "raw":
            # DTW only makes sense on time-aligned signatures; for non-raw
            # signatures we transparently fall back to Euclidean to keep
            # the ablation matrix well-defined.
            D = euclidean_distance_matrix(S)
        else:
            D = dtw_distance_matrix(S)
        return cluster_with_min_size(D, k, min_size=3, seed=seed)
    if algorithm == "dtw_kmedoids":
        D = dtw_distance_matrix(S) if signature_name == "raw" else euclidean_distance_matrix(S)
        labels = cluster_kmedoids(D, k, seed=seed)
        return labels
    if algorithm == "kshape":
        return cluster_kshape(X_train, k, seed=seed)
    if algorithm == "euclidean_ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(S)
    raise ValueError(f"Unknown algorithm {algorithm}")


# ---------------------------------------------------------------------------
# Ablation A -- signature
# ---------------------------------------------------------------------------


def ablation_signature(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int = ABLATION_HORIZON,
) -> pd.DataFrame:
    rows = []
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    for sig_name in ("raw", "weekly_profile", "spectral_envelope", "summary_stats"):
        labels_per_window = []
        mapping_per_window = []
        for test_start, _ in test_windows:
            X_train = X[:, :test_start]
            labels = build_partition(X_train, sig_name, "dtw_agglomerative", k=4)
            mapping = {
                int(c): _occam_select_for_cluster(
                    X_train, labels == c, horizon, start_date, ("catboost", "ridge")
                )
                for c in np.unique(labels)
            }
            labels_per_window.append(labels)
            mapping_per_window.append(mapping)
        result = evaluate_partition(
            dataset_name, X, start_date, horizon, labels_per_window, mapping_per_window
        )
        rows.append({"dataset": dataset_name, "horizon": horizon, "signature": sig_name, **result})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation B -- clustering algorithm
# ---------------------------------------------------------------------------


def ablation_algorithm(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int = ABLATION_HORIZON,
) -> pd.DataFrame:
    rows = []
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    for alg in ("dtw_agglomerative", "dtw_kmedoids", "kshape", "euclidean_ward"):
        labels_per_window = []
        mapping_per_window = []
        for test_start, _ in test_windows:
            X_train = X[:, :test_start]
            labels = build_partition(X_train, "raw", alg, k=4)
            mapping = {
                int(c): _occam_select_for_cluster(
                    X_train, labels == c, horizon, start_date, ("catboost", "ridge")
                )
                for c in np.unique(labels)
            }
            labels_per_window.append(labels)
            mapping_per_window.append(mapping)
        result = evaluate_partition(
            dataset_name, X, start_date, horizon, labels_per_window, mapping_per_window
        )
        rows.append({"dataset": dataset_name, "horizon": horizon, "algorithm": alg, **result})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation C -- selection strategy
# ---------------------------------------------------------------------------


def ablation_selection(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int = ABLATION_HORIZON,
) -> pd.DataFrame:
    rows = []
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    strategies = {
        "catboost_only": lambda *args: "catboost",
        "ridge_only": lambda *args: "ridge",
        "cv_best": lambda X_train, mask, h, sd: _occam_select_for_cluster(
            X_train, mask, h, sd, ("catboost", "ridge")
        ) if False else _cv_best(X_train, mask, h, sd),
        "cv_occam": lambda X_train, mask, h, sd: _occam_select_for_cluster(
            X_train, mask, h, sd, ("catboost", "ridge")
        ),
    }
    for strat_name, strat_fn in strategies.items():
        labels_per_window = []
        mapping_per_window = []
        for test_start, _ in test_windows:
            X_train = X[:, :test_start]
            labels = build_partition(X_train, "raw", "dtw_agglomerative", k=4)
            mapping = {
                int(c): strat_fn(X_train, labels == c, horizon, start_date)
                for c in np.unique(labels)
            }
            labels_per_window.append(labels)
            mapping_per_window.append(mapping)
        result = evaluate_partition(
            dataset_name, X, start_date, horizon, labels_per_window, mapping_per_window
        )
        rows.append({"dataset": dataset_name, "horizon": horizon, "strategy": strat_name, **result})
    return pd.DataFrame(rows)


def _cv_best(X_train: np.ndarray, mask: np.ndarray, horizon: int, start_date: str) -> str:
    """CV-best without Occam: pure argmin of inner CV MAE."""
    from forecast import _inner_cv_mae

    member_ids = list(np.where(mask)[0])
    cluster_X = X_train[member_ids]
    scores = {
        m: _inner_cv_mae(m, cluster_X, member_ids, start_date, horizon)
        for m in ("catboost", "ridge")
    }
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        return "catboost"
    return min(finite, key=finite.get)


# ---------------------------------------------------------------------------
# Ablation D -- number of clusters
# ---------------------------------------------------------------------------


def ablation_k(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int = ABLATION_HORIZON,
) -> pd.DataFrame:
    rows = []
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    for k in (2, 3, 4, 5, 6):
        labels_per_window = []
        mapping_per_window = []
        for test_start, _ in test_windows:
            X_train = X[:, :test_start]
            labels = build_partition(X_train, "raw", "dtw_agglomerative", k=k)
            mapping = {
                int(c): _occam_select_for_cluster(
                    X_train, labels == c, horizon, start_date, ("catboost", "ridge")
                )
                for c in np.unique(labels)
            }
            labels_per_window.append(labels)
            mapping_per_window.append(mapping)
        result = evaluate_partition(
            dataset_name, X, start_date, horizon, labels_per_window, mapping_per_window
        )
        rows.append(
            {
                "dataset": dataset_name,
                "horizon": horizon,
                "k_requested": k,
                "k_actual": int(labels.max() + 1),
                **result,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Occam tau sensitivity
# ---------------------------------------------------------------------------


def ablation_tau(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int = ABLATION_HORIZON,
) -> pd.DataFrame:
    """Sweep the Occam tolerance threshold; small clusters share the
    same value as large ones in this study to keep the axis 1-D."""
    import forecast as fc

    rows = []
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    saved_large, saved_small = fc.OCCAM_TAU_LARGE, fc.OCCAM_TAU_SMALL
    try:
        for tau in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40):
            fc.OCCAM_TAU_LARGE = tau
            fc.OCCAM_TAU_SMALL = tau
            labels_per_window = []
            mapping_per_window = []
            ridge_share_per_window = []
            for test_start, _ in test_windows:
                X_train = X[:, :test_start]
                labels = build_partition(X_train, "raw", "dtw_agglomerative", k=4)
                mapping = {
                    int(c): _occam_select_for_cluster(
                        X_train, labels == c, horizon, start_date, ("catboost", "ridge")
                    )
                    for c in np.unique(labels)
                }
                ridge_share_per_window.append(
                    sum(1 for v in mapping.values() if v == "ridge") / max(1, len(mapping))
                )
                labels_per_window.append(labels)
                mapping_per_window.append(mapping)
            result = evaluate_partition(
                dataset_name, X, start_date, horizon, labels_per_window, mapping_per_window
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "tau": tau,
                    "mean_ridge_share": float(np.mean(ridge_share_per_window)),
                    **result,
                }
            )
    finally:
        fc.OCCAM_TAU_LARGE = saved_large
        fc.OCCAM_TAU_SMALL = saved_small
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ABLATIONS = [
    ("ablation_A_signature", ablation_signature),
    ("ablation_B_algorithm", ablation_algorithm),
    ("ablation_C_selection", ablation_selection),
    ("ablation_D_k", ablation_k),
    ("ablation_tau", ablation_tau),
]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for dataset_name in ("ELEC", "NN5"):
        print(f"\n=== Loading {dataset_name} ===")
        X, _, start_date = load_dataset(dataset_name)
        if X is None:
            print(f"  [!] {dataset_name} unavailable; skipping")
            continue
        for label, fn in ABLATIONS:
            print(f"--- {label} on {dataset_name} ---")
            df = fn(dataset_name, X, start_date)
            out_path = os.path.join(RESULTS_DIR, f"{label}.csv")
            mode = "a" if os.path.exists(out_path) else "w"
            header = mode == "w"
            df.to_csv(out_path, mode=mode, header=header, index=False)
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
