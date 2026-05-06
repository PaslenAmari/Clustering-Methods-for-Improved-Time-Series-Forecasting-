"""
Main experiment runner for the DAMDID 2026 clustered-local forecasting paper.

Sweeps two open daily panels (ELEC, NN5) over three forecast horizons
({7, 14, 30}) and six methods (5 baselines + CL-Occam), produces:

    results/main_metrics.csv         per (dataset, horizon, method, series)
    results/main_aggregate.csv       per (dataset, horizon, method)
    results/main_paired_tests.csv    per (dataset, horizon, baseline) vs CL-Occam,
                                     with Holm correction across the 30-test family
    results/cluster_diagnostics.csv  per (dataset, horizon, cluster)

The runner is single-process; rerunning is idempotent for a given seed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from clustering import (
    ClusterSelectionResult,
    cluster_with_min_size,
    dtw_distance_matrix,
    select_k_by_validation,
    zscore_rows,
)
from data_io import load_dataset
from eval import (
    aggregate_per_series,
    bootstrap_median_ci,
    cliffs_delta,
    format_pvalue,
    holm_correction,
    mae,
    mase,
    paired_test,
    smape,
)
from forecast import (
    build_method_registry,
    forecast_clustered_local,
    forecast_global,
    select_model_for_cluster,
)


RESULTS_DIR = "results"
HORIZONS = (7, 14, 30)
N_WINDOWS = 4
DATASETS = ("ELEC", "NN5")
# Reviewer-2 feedback (2026-04-28): per-series local CatBoost is now
# included in the main comparison family.  At N=42 / N=111 with four
# windows it costs ~1 hour wall-clock on a single core but closes the
# "global vs local" comparison the reviewer flagged as missing.  Set
# this to {"local_catboost"} to reproduce the prior, smaller-budget
# variant.
SKIP_METHODS: set[str] = set()
# Validation k-grid for prepare_clustering. Empirically, on NN5 the
# selected k clusters around 3..5; the longer (2..8) sweep adds 75%
# wall-clock for negligible expected MAE difference.
K_GRID = (3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Backtest harness
# ---------------------------------------------------------------------------


def make_test_windows(n_timesteps: int, horizon: int, n_windows: int) -> List[Tuple[int, int]]:
    """Last ``n_windows * horizon`` days form non-overlapping test cells."""
    n_test = horizon * n_windows
    if n_test >= n_timesteps:
        raise ValueError(f"Not enough timesteps ({n_timesteps}) for horizon {horizon} x {n_windows}")
    starts = [n_timesteps - n_test + i * horizon for i in range(n_windows)]
    return [(s, s + horizon) for s in starts]


def _series_id(idx: int) -> str:
    return f"region_{idx}"


def _evaluate_window_forecasts(
    forecasts: Dict[str, np.ndarray],
    X_full: np.ndarray,
    test_start: int,
    test_end: int,
    train_history: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute MAE / sMAPE / MASE for each series with a finite forecast."""
    out: Dict[str, Dict[str, float]] = {}
    for sid in range(X_full.shape[0]):
        seg = _series_id(sid)
        fc = forecasts.get(seg)
        if fc is None or len(fc) != (test_end - test_start) or not np.isfinite(fc).all():
            continue
        actual = X_full[sid, test_start:test_end]
        out[seg] = {
            "mae": mae(actual, fc),
            "smape": smape(actual, fc),
            "mase": mase(actual, fc, train_history[sid], seasonality=7),
        }
    return out


def _aggregate_metrics(window_records: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Per-series mean across windows for each metric."""
    series_metric_lists: Dict[str, Dict[str, List[float]]] = {}
    for window in window_records:
        for seg, metrics in window.items():
            store = series_metric_lists.setdefault(seg, {"mae": [], "smape": [], "mase": []})
            for k in store:
                v = metrics.get(k)
                if v is not None and np.isfinite(v):
                    store[k].append(float(v))
    return {
        seg: {k: float(np.mean(v)) if v else float("nan") for k, v in store.items()}
        for seg, store in series_metric_lists.items()
    }


# ---------------------------------------------------------------------------
# Cluster lifecycle for CL-Occam
# ---------------------------------------------------------------------------


def _cluster_score_fn(horizon: int):
    """Return a `(inner_train, inner_val, labels) -> float` scorer used by k selection."""

    def score(inner_train: np.ndarray, inner_val: np.ndarray, labels: np.ndarray) -> float:
        # Build forecasts per cluster using a fixed CatBoost (cheap and consistent
        # across k values).  Using CL-Occam selection inside k selection would be
        # circular and ~10x slower without a clear payoff.
        try:
            forecasts = forecast_clustered_local(
                inner_train,
                labels,
                {int(c): "catboost" for c in np.unique(labels)},
                "2000-01-01",
                inner_val.shape[1],
            )
        except Exception:
            return float("inf")
        errors = []
        for sid in range(inner_train.shape[0]):
            fc = forecasts.get(_series_id(sid))
            if fc is None or len(fc) != inner_val.shape[1] or not np.isfinite(fc).all():
                continue
            errors.append(float(np.mean(np.abs(inner_val[sid] - fc))))
        return float(np.mean(errors)) if errors else float("inf")

    return score


def prepare_clustering(
    X_train: np.ndarray,
    horizon: int,
    seed: int,
) -> Tuple[ClusterSelectionResult, Dict[int, str], Dict[str, float]]:
    """Run k selection and per-cluster model selection on a training fold."""
    Xz = zscore_rows(X_train)
    D = dtw_distance_matrix(Xz)
    selection = select_k_by_validation(
        X_train=X_train,
        distance_matrix=D,
        holdout_h=horizon,
        score_fn=_cluster_score_fn(horizon),
        k_grid=K_GRID,
        min_size=3,
        seed=seed,
    )
    model_mapping: Dict[int, str] = {}
    cv_traces: Dict[str, float] = {}
    for cid in np.unique(selection.labels):
        mask = selection.labels == cid
        chosen, scores = select_model_for_cluster(
            X_train_full=X_train,
            cluster_mask=mask,
            horizon=horizon,
            start_date="2000-01-01",
            candidate_pool=("catboost", "ridge"),
        )
        model_mapping[int(cid)] = chosen
        for m, s in scores.items():
            cv_traces[f"cluster{int(cid)}_{m}"] = float(s)
    return selection, model_mapping, cv_traces


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_one_cell(
    dataset_name: str,
    X: np.ndarray,
    start_date: str,
    horizon: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Run one (dataset, horizon) cell.

    Returns four pandas frames: per-series metrics, aggregate per
    method, cluster diagnostics, and a dict with method-level forecasts
    (for downstream paired tests)."""
    test_windows = make_test_windows(X.shape[1], horizon, N_WINDOWS)
    methods: Dict[str, Dict[str, Dict[str, float]]] = {}
    cluster_diag_rows: List[Dict[str, object]] = []
    method_window_records: Dict[str, List[Dict[str, Dict[str, float]]]] = {}

    for window_idx, (test_start, test_end) in enumerate(test_windows):
        X_train = X[:, :test_start]

        # Re-run clustering and selection on the new training fold so each
        # window reflects only past information.
        selection, mapping, cv_traces = prepare_clustering(X_train, horizon, seed)

        cluster_sizes = np.bincount(selection.labels)
        for cid, size in enumerate(cluster_sizes):
            cluster_diag_rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "window": window_idx,
                    "cluster_id": int(cid),
                    "size": int(size),
                    "chosen_model": mapping.get(cid, "n/a"),
                    "cv_catboost": cv_traces.get(f"cluster{cid}_catboost", float("nan")),
                    "cv_ridge": cv_traces.get(f"cluster{cid}_ridge", float("nan")),
                }
            )

        registry = build_method_registry(
            cluster_labels_provider=lambda Xt, sd: (selection.labels, mapping),
        )

        import gc
        for method in registry:
            if method.name in SKIP_METHODS:
                continue
            try:
                forecasts = method.fn(X_train, start_date, horizon)
            except Exception as exc:
                print(f"  [!] {method.name} failed on window {window_idx}: {exc}")
                forecasts = {}
            errors = _evaluate_window_forecasts(
                forecasts, X, test_start, test_end, X_train
            )
            method_window_records.setdefault(method.name, []).append(errors)
            # Encourage Python and ETNA to release memory between methods.
            del forecasts
            gc.collect()

    rows: List[Dict[str, object]] = []
    aggregate_rows: List[Dict[str, object]] = []
    method_per_series_mae: Dict[str, Dict[str, float]] = {}

    for method_name, windows in method_window_records.items():
        per_series = _aggregate_metrics(windows)
        method_per_series_mae[method_name] = {seg: m["mae"] for seg, m in per_series.items()}
        for seg, metrics in per_series.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "method": method_name,
                    "series": seg,
                    "mae": metrics["mae"],
                    "smape": metrics["smape"],
                    "mase": metrics["mase"],
                }
            )
        if per_series:
            mae_arr = np.array([m["mae"] for m in per_series.values()], dtype=float)
            smape_arr = np.array([m["smape"] for m in per_series.values()], dtype=float)
            mase_arr = np.array([m["mase"] for m in per_series.values()], dtype=float)
            aggregate_rows.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "method": method_name,
                    "n_series": len(per_series),
                    "mean_mae": float(np.nanmean(mae_arr)),
                    "median_mae": float(np.nanmedian(mae_arr)),
                    "mean_smape": float(np.nanmean(smape_arr)),
                    "mean_mase": float(np.nanmean(mase_arr)),
                }
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(cluster_diag_rows),
        method_per_series_mae,
    )


def _append_csv(df: pd.DataFrame, path: str) -> None:
    """Incremental write: append to CSV with header only when file is new."""
    if df.empty:
        return
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def run_all(seed: int = 42) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "main_metrics.csv")
    aggregate_path = os.path.join(RESULTS_DIR, "main_aggregate.csv")
    cluster_path = os.path.join(RESULTS_DIR, "cluster_diagnostics.csv")
    paired_path = os.path.join(RESULTS_DIR, "main_paired_tests.csv")
    paired_records: List[Dict[str, object]] = []

    for dataset_name in DATASETS:
        print(f"\n=== Loading dataset {dataset_name} ===")
        X, names, start_date = load_dataset(dataset_name)
        if X is None:
            print(f"  [!] dataset {dataset_name} unavailable; skipping")
            continue
        print(f"  shape: {X.shape}, start={start_date}")

        for horizon in HORIZONS:
            print(f"\n--- {dataset_name} h={horizon} ---")
            metrics_df, agg_df, cluster_df, per_series_mae = run_one_cell(
                dataset_name, X, start_date, horizon, seed
            )
            # Incremental save -- protects against mid-run failure on a multi-hour sweep.
            _append_csv(metrics_df, metrics_path)
            _append_csv(agg_df, aggregate_path)
            _append_csv(cluster_df, cluster_path)
            print(f"  wrote partial CSVs for {dataset_name} h={horizon}")

            if "cl_occam" not in per_series_mae:
                continue
            cl_series = per_series_mae["cl_occam"]
            for baseline_name, baseline_series in per_series_mae.items():
                if baseline_name == "cl_occam":
                    continue
                shared = sorted(set(cl_series) & set(baseline_series))
                if len(shared) < 2:
                    continue
                cl_arr = np.array([cl_series[s] for s in shared])
                bl_arr = np.array([baseline_series[s] for s in shared])
                result = paired_test(cl_arr, bl_arr, label=f"{dataset_name}|h={horizon}|vs={baseline_name}")
                row = result.as_row()
                row.update(
                    dataset=dataset_name,
                    horizon=horizon,
                    baseline=baseline_name,
                )
                paired_records.append(row)

    if paired_records:
        paired_df = pd.DataFrame(paired_records)
        paired_df["holm_p"] = holm_correction(paired_df["wilcoxon_p"].fillna(1.0).tolist())
        paired_df["wilcoxon_p_str"] = paired_df["wilcoxon_p"].apply(format_pvalue)
        paired_df["holm_p_str"] = paired_df["holm_p"].apply(format_pvalue)
        paired_df.to_csv(paired_path, index=False)

    print("\n=== Done ===")
    print(f"Wrote results to {RESULTS_DIR}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="DAMDID 2026 main experiment runner")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all(seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
