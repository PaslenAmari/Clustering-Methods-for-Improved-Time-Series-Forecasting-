"""Rossmann sweep with exogenous features (Promo, StateHoliday, SchoolHoliday, Open).

This is a self-contained variant of the main pipeline that wires the
Kaggle Rossmann exogenous columns into the ETNA TSDataset, so the
gradient boosters and Ridge variants get to see the holiday/promo
signal that top Kaggle solutions exploit heavily.  Naive-7 and ETS
ignore exog by construction.

Method order matches the main paper (skipping local_catboost as on
the no-exog Rossmann run).  We deliberately keep CatBoost-matched
hyperparameters and the same lag/calendar feature stack so the
Rossmann-with-exog comparison stays as parallel as possible to the
without-exog baseline.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import (
    DateFlagsTransform,
    LagTransform,
    MeanSegmentEncoderTransform,
    SegmentEncoderTransform,
    StandardScalerTransform,
    TimeSeriesImputerTransform,
)

import run_damdid
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
from forecast import CATBOOST_PARAMS, DEFAULT_LAGS, RIDGE_ALPHA, make_model


ROSSMANN_RAW_PATH = os.path.join("..", "data", "rossmann", "train.csv")
ROSSMANN_FULL_LENGTH = 942
ROSSMANN_TARGET_N = 50
ROSSMANN_SEED = 42
HORIZONS = (7, 14, 30)
N_WINDOWS = 4
EXOG_COLS = ("Promo", "StateHoliday", "SchoolHoliday", "Open")
DATASET_NAME = "ROSSMANN_EXOG"


def _load_with_exog() -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    df = pd.read_csv(
        ROSSMANN_RAW_PATH,
        parse_dates=["Date"],
        dtype={"Store": "int32"},
    )
    df["StateHoliday"] = df["StateHoliday"].astype(str).map(
        {"0": 0, "a": 1, "b": 2, "c": 3}
    ).fillna(0).astype(int)
    df = df.sort_values(["Store", "Date"])
    counts = df.groupby("Store").size()
    full_stores = counts[counts == ROSSMANN_FULL_LENGTH].index.tolist()
    rng = np.random.default_rng(ROSSMANN_SEED)
    chosen = sorted(rng.choice(full_stores, size=ROSSMANN_TARGET_N, replace=False).tolist())
    sub = df[df["Store"].isin(chosen)].copy()

    target_long = sub.rename(
        columns={"Sales": "target", "Date": "timestamp", "Store": "segment"}
    )[["timestamp", "segment", "target"]]
    target_long["segment"] = target_long["segment"].astype(str)

    exog_long = sub.rename(
        columns={"Date": "timestamp", "Store": "segment"}
    )[["timestamp", "segment"] + list(EXOG_COLS)]
    exog_long["segment"] = exog_long["segment"].astype(str)

    return target_long, exog_long, chosen


def _build_tsdataset(
    target_long: pd.DataFrame, exog_long: pd.DataFrame
) -> TSDataset:
    target_wide = TSDataset.to_dataset(target_long)
    exog_wide = TSDataset.to_dataset(exog_long)
    return TSDataset(
        df=target_wide,
        df_exog=exog_wide,
        freq="D",
        known_future=list(EXOG_COLS),
    )


def _build_transforms(model_type: str) -> list:
    transforms = [
        TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
        LagTransform(in_column="target", lags=list(DEFAULT_LAGS)),
        DateFlagsTransform(day_number_in_week=True, is_weekend=True),
        StandardScalerTransform(in_column="target"),
    ]
    if model_type == "catboost_id":
        transforms.append(SegmentEncoderTransform())
    elif model_type == "ridge_id":
        transforms.append(MeanSegmentEncoderTransform())
    return transforms


def _build_target_matrix(target_long: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    pivot = target_long.pivot(index="timestamp", columns="segment", values="target")
    pivot = pivot.sort_index()
    segs = list(pivot.columns)
    matrix = pivot.T.values.astype(float)
    return matrix, segs


def _cl_occam_clustering(
    train_target: pd.DataFrame, horizon: int, seed: int = 42
):
    """Run DTW clustering + per-cluster CV-Occam selection (target only).

    Returns ``(labels, mapping, segs)`` where labels[i] is the cluster
    of segs[i] and mapping[c] is the chosen model class for cluster c.
    """
    from clustering import (
        ClusterSelectionResult,
        dtw_distance_matrix,
        select_k_by_validation,
        zscore_rows,
    )
    from forecast import select_model_for_cluster, forecast_clustered_local

    matrix, segs = _build_target_matrix(train_target)
    Xz = zscore_rows(matrix)
    D = dtw_distance_matrix(Xz)

    def score_fn(inner_train, inner_val, labels):
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
        errs = []
        for sid in range(inner_train.shape[0]):
            seg_name = f"region_{sid}"
            fc = forecasts.get(seg_name)
            if fc is None or len(fc) != inner_val.shape[1] or not np.isfinite(fc).all():
                continue
            errs.append(float(np.mean(np.abs(inner_val[sid] - fc))))
        return float(np.mean(errs)) if errs else float("inf")

    selection = select_k_by_validation(
        X_train=matrix,
        distance_matrix=D,
        holdout_h=horizon,
        score_fn=score_fn,
        k_grid=(3, 4, 5, 6),
        min_size=3,
        seed=seed,
    )
    mapping = {}
    for cid in np.unique(selection.labels):
        mask = selection.labels == cid
        chosen, _ = select_model_for_cluster(
            X_train_full=matrix,
            cluster_mask=mask,
            horizon=horizon,
            start_date="2000-01-01",
            candidate_pool=("catboost", "ridge"),
        )
        mapping[int(cid)] = chosen
    return selection.labels, mapping, segs


def _forecast_cl_occam_with_exog(
    target_long: pd.DataFrame,
    exog_long: pd.DataFrame,
    horizon: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """CL-Occam fit per-cluster with the Rossmann exog regressors."""
    labels, mapping, segs = _cl_occam_clustering(target_long, horizon, seed=seed)
    out: Dict[str, np.ndarray] = {}
    for cid in np.unique(labels):
        member_segs = [segs[i] for i in range(len(segs)) if labels[i] == cid]
        model_class = mapping[int(cid)]
        target_sub = target_long[target_long["segment"].isin(member_segs)]
        exog_sub = exog_long[exog_long["segment"].isin(member_segs)]
        tsd = _build_tsdataset(target_sub, exog_sub)
        transforms = _build_transforms(model_class)
        pipe = Pipeline(
            model=make_model(model_class), transforms=transforms, horizon=horizon
        )
        try:
            pipe.fit(tsd)
            fdf = pipe.forecast().to_pandas()
            for seg in fdf.columns.get_level_values(0).unique():
                out[seg] = fdf[seg]["target"].values
        except Exception as exc:
            print(f"  [!] cl_occam cluster {cid} ({model_class}) failed: {exc}", flush=True)
            continue
    return out


def _forecast_method(
    model_type: str, train_tsd: TSDataset, horizon: int
) -> Dict[str, np.ndarray]:
    if model_type == "naive_seasonal_7":
        train_df = train_tsd.to_pandas()
        out = {}
        for seg in train_df.columns.get_level_values(0).unique():
            history = train_df[seg]["target"].values
            if len(history) < 7:
                continue
            last_cycle = history[-7:]
            reps = int(np.ceil(horizon / 7))
            out[seg] = np.tile(last_cycle, reps)[:horizon]
        return out
    if model_type == "ets":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        train_df = train_tsd.to_pandas()
        out = {}
        for seg in train_df.columns.get_level_values(0).unique():
            history = train_df[seg]["target"].values
            if len(history) < 24:
                continue
            try:
                m = ExponentialSmoothing(
                    history, trend="add", seasonal="add",
                    seasonal_periods=7, initialization_method="estimated"
                ).fit()
                fc = np.asarray(m.forecast(horizon), dtype=float)
                if not np.isfinite(fc).all():
                    last_cycle = history[-7:]
                    reps = int(np.ceil(horizon / 7))
                    fc = np.tile(last_cycle, reps)[:horizon]
                out[seg] = fc
            except Exception:
                last_cycle = history[-7:]
                reps = int(np.ceil(horizon / 7))
                out[seg] = np.tile(last_cycle, reps)[:horizon]
        return out
    transforms = _build_transforms(model_type)
    pipe = Pipeline(
        model=make_model(model_type),
        transforms=transforms,
        horizon=horizon,
    )
    pipe.fit(train_tsd)
    fdf = pipe.forecast().to_pandas()
    return {
        seg: fdf[seg]["target"].values
        for seg in fdf.columns.get_level_values(0).unique()
    }


def _evaluate(
    forecasts: Dict[str, np.ndarray],
    test_long: pd.DataFrame,
    train_history: Dict[str, np.ndarray],
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for seg, fc in forecasts.items():
        actual_rows = test_long[test_long["segment"] == seg].sort_values("timestamp")
        if len(actual_rows) != horizon:
            continue
        actual = actual_rows["target"].values.astype(float)
        if not (np.isfinite(fc).all() and np.isfinite(actual).all()):
            continue
        history = train_history.get(seg)
        if history is None:
            continue
        out[seg] = {
            "mae": mae(actual, fc),
            "smape": smape(actual, fc),
            "mase": mase(actual, fc, history, seasonality=7),
        }
    return out


def _aggregate_metrics(window_records: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    series_metric_lists = {}
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


def main() -> int:
    metrics_path = os.path.join("results", "main_metrics.csv")
    aggregate_path = os.path.join("results", "main_aggregate.csv")

    print("=== Loading ROSSMANN with exog ===", flush=True)
    target_long, exog_long, stores = _load_with_exog()
    timestamps = sorted(target_long["timestamp"].unique())
    print(f"  stores={len(stores)}, days={len(timestamps)}", flush=True)

    methods = (
        "naive_seasonal_7",
        "ets",
        "catboost",
        "ridge",
        "cl_occam",
    )
    method_label = {
        "naive_seasonal_7": "naive_seasonal_7",
        "ets": "ets",
        "catboost": "global_catboost",
        "ridge": "global_ridge",
        "cl_occam": "cl_occam",
    }

    new_metrics_rows = []
    new_agg_rows = []

    for horizon in HORIZONS:
        print(f"\n--- {DATASET_NAME} h={horizon} ---", flush=True)
        n_test = horizon * N_WINDOWS
        if len(timestamps) <= n_test + 30:
            continue
        starts = [len(timestamps) - n_test + i * horizon for i in range(N_WINDOWS)]

        method_window_records = {m: [] for m in methods}
        for window_idx, test_start in enumerate(starts):
            test_end = test_start + horizon
            train_ts_dates = timestamps[:test_start]
            test_ts_dates = timestamps[test_start:test_end]

            train_target = target_long[target_long["timestamp"].isin(train_ts_dates)].copy()
            train_exog = exog_long[exog_long["timestamp"].isin(timestamps[:test_end])].copy()
            tsd_train = _build_tsdataset(train_target, train_exog)

            test_target = target_long[target_long["timestamp"].isin(test_ts_dates)].copy()
            train_history = {
                seg: train_target[train_target["segment"] == seg].sort_values("timestamp")["target"].values.astype(float)
                for seg in train_target["segment"].unique()
            }

            for m in methods:
                try:
                    if m == "cl_occam":
                        train_exog_for_cluster = exog_long[
                            exog_long["timestamp"].isin(timestamps[:test_end])
                        ]
                        forecasts = _forecast_cl_occam_with_exog(
                            train_target, train_exog_for_cluster, horizon
                        )
                    else:
                        forecasts = _forecast_method(m, tsd_train, horizon)
                except Exception as exc:
                    print(f"  [!] {m} failed window {window_idx}: {exc}", flush=True)
                    forecasts = {}
                errs = _evaluate(forecasts, test_target, train_history, horizon)
                method_window_records[m].append(errs)

        for m, windows in method_window_records.items():
            per_series = _aggregate_metrics(windows)
            method_name = method_label[m]
            for seg, metrics in per_series.items():
                new_metrics_rows.append({
                    "dataset": DATASET_NAME, "horizon": horizon, "method": method_name,
                    "series": f"region_{seg}", "mae": metrics["mae"],
                    "smape": metrics["smape"], "mase": metrics["mase"],
                })
            if per_series:
                mae_arr = np.array([m_["mae"] for m_ in per_series.values()], dtype=float)
                smape_arr = np.array([m_["smape"] for m_ in per_series.values()], dtype=float)
                mase_arr = np.array([m_["mase"] for m_ in per_series.values()], dtype=float)
                new_agg_rows.append({
                    "dataset": DATASET_NAME, "horizon": horizon, "method": method_name,
                    "n_series": len(per_series),
                    "mean_mae": float(np.nanmean(mae_arr)),
                    "median_mae": float(np.nanmedian(mae_arr)),
                    "mean_smape": float(np.nanmean(smape_arr)),
                    "mean_mase": float(np.nanmean(mase_arr)),
                })
        print(f"  wrote {DATASET_NAME} h={horizon}", flush=True)

    if new_metrics_rows:
        m_df = pd.DataFrame(new_metrics_rows)
        m_df.to_csv(metrics_path, mode="a", header=False, index=False)
        a_df = pd.DataFrame(new_agg_rows)
        a_df.to_csv(aggregate_path, mode="a", header=False, index=False)
        print(f"\n=== {DATASET_NAME} Done ===", flush=True)
    else:
        print("[!] no rows produced", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
