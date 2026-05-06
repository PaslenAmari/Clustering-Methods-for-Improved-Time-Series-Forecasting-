"""
Forecasting models, baselines, and the per-cluster CV-Occam selector
for the DAMDID 2026 clustered-local pipeline.

Two design constraints that distinguish this module from the prior
implementation:

* Identical hyperparameters across the global, local, and cluster
  branches whenever a model class is reused.  This isolates the
  clustering and selection effects from regularization differences
  that previously confounded the comparison.
* All baselines and the proposed method share the same feature stack
  (lags, calendar flags) and the same backtest harness.  Performance
  differences therefore reflect the modelling decision under study,
  not feature engineering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from etna.datasets import TSDataset
from etna.models import CatBoostMultiSegmentModel, SklearnMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import (
    DateFlagsTransform,
    LagTransform,
    MeanSegmentEncoderTransform,
    SegmentEncoderTransform,
    StandardScalerTransform,
    TimeSeriesImputerTransform,
)


# ---------------------------------------------------------------------------
# Hyperparameter contracts
# ---------------------------------------------------------------------------

_USE_GPU = os.environ.get("CL_OCCAM_CATBOOST_GPU", "0") == "1"

CATBOOST_PARAMS = dict(
    iterations=300,
    depth=5,
    learning_rate=0.03,
    random_seed=42,
    logging_level="Silent",
    **({"task_type": "GPU", "devices": "0"} if _USE_GPU else {"thread_count": -1}),
)
"""Single CatBoost hyperparameter set used by global, local and per-cluster
branches.  Set the env var ``CL_OCCAM_CATBOOST_GPU=1`` to switch to GPU
training.  The CPU default uses every available core
(``thread_count=-1``).  GPU is recommended when the panel has a single
heavy global fit per window (Global CB / CB+ID / CL-Occam-with-CatBoost
clusters) and is *not* recommended for ``local_catboost`` where the
$\\sim{}300$ small fits per cell are dominated by per-fit GPU
initialisation overhead."""

RIDGE_ALPHA = 0.5

DEFAULT_LAGS = [1, 2, 3, 7, 14, 30]
DEFAULT_FREQ = "D"


# ---------------------------------------------------------------------------
# ETNA helpers
# ---------------------------------------------------------------------------


def make_ts_dataset(
    X_sub: np.ndarray,
    segment_ids: Sequence[int],
    start_date: str,
    freq: str = DEFAULT_FREQ,
) -> TSDataset:
    timestamps = pd.date_range(start_date, periods=X_sub.shape[1], freq=freq)
    rows = []
    for i, sid in enumerate(segment_ids):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "segment": f"region_{sid}",
                    "target": X_sub[i],
                }
            )
        )
    return TSDataset(TSDataset.to_dataset(pd.concat(rows, ignore_index=True)), freq=freq)


def build_transforms(with_imputer: bool, lags: Sequence[int] = DEFAULT_LAGS):
    transforms = []
    if with_imputer:
        transforms.append(TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"))
    transforms.extend(
        [
            LagTransform(in_column="target", lags=list(lags)),
            DateFlagsTransform(day_number_in_week=True, is_weekend=True),
            StandardScalerTransform(in_column="target"),
        ]
    )
    return transforms


def make_model(model_type: str):
    if model_type == "catboost":
        return CatBoostMultiSegmentModel(**CATBOOST_PARAMS)
    if model_type == "catboost_id":
        # CatBoost with native categorical handling of the series identifier.
        # SegmentEncoderTransform adds a 'segment_code' column with
        # ``dtype='category'``; CatBoost auto-detects categorical features
        # from dtype (the same auto-detection already covers the
        # DateFlags-generated columns in the baseline pipeline), so we do
        # not need to pass ``cat_features`` explicitly here.  Passing it
        # would override auto-detection and crash on the DateFlags columns.
        return CatBoostMultiSegmentModel(**CATBOOST_PARAMS)
    if model_type == "ridge":
        return SklearnMultiSegmentModel(
            regressor=make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0),
                Ridge(alpha=RIDGE_ALPHA, random_state=42),
            )
        )
    if model_type == "ridge_id":
        # Linear analogue of the ID baseline: a single per-segment target
        # mean column added by MeanSegmentEncoderTransform acts as an
        # entity-specific intercept.  Equivalent in spirit to a one-hot
        # expansion for a Ridge regressor without N-1 extra columns.
        return SklearnMultiSegmentModel(
            regressor=make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0),
                Ridge(alpha=RIDGE_ALPHA, random_state=42),
            )
        )
    raise ValueError(f"Unknown model type {model_type!r}")


def _forecast_with(
    model_type: str,
    X_sub: np.ndarray,
    segment_ids: Sequence[int],
    start_date: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Run a single ETNA pipeline and return {segment_name: forecast}."""
    ts = make_ts_dataset(X_sub, segment_ids, start_date)
    transforms = build_transforms(
        with_imputer=(model_type in {"ridge", "ridge_id"}),
    )
    if model_type == "catboost_id":
        transforms.append(SegmentEncoderTransform())
    elif model_type == "ridge_id":
        transforms.append(MeanSegmentEncoderTransform())
    pipe = Pipeline(
        model=make_model(model_type),
        transforms=transforms,
        horizon=horizon,
    )
    pipe.fit(ts)
    fdf = pipe.forecast().to_pandas()
    return {seg: fdf[seg]["target"].values for seg in fdf.columns.get_level_values(0).unique()}


# ---------------------------------------------------------------------------
# Per-cluster CV-Occam selector
# ---------------------------------------------------------------------------


OCCAM_TAU_LARGE = 0.10
"""Robustness margin for clusters of size >= 4."""

OCCAM_TAU_SMALL = 0.25
"""Robustness margin for clusters smaller than 4."""


def _inner_cv_mae(
    model_type: str,
    X_train_cluster: np.ndarray,
    member_ids: Sequence[int],
    start_date: str,
    horizon: int,
) -> float:
    """Inner CV MAE: train on cluster minus last horizon, score on the held-out tail."""
    if X_train_cluster.shape[1] <= horizon + 30:
        return float("inf")
    inner_train = X_train_cluster[:, :-horizon]
    inner_val = X_train_cluster[:, -horizon:]
    try:
        forecasts = _forecast_with(model_type, inner_train, member_ids, start_date, horizon)
    except Exception:
        return float("inf")
    errors = []
    for local_i, sid in enumerate(member_ids):
        seg = f"region_{sid}"
        fc = forecasts.get(seg)
        if fc is None or len(fc) != horizon or not np.isfinite(fc).all():
            continue
        errors.append(float(np.mean(np.abs(inner_val[local_i] - fc))))
    return float(np.mean(errors)) if errors else float("inf")


def select_model_for_cluster(
    X_train_full: np.ndarray,
    cluster_mask: np.ndarray,
    horizon: int,
    start_date: str,
    candidate_pool: Sequence[str] = ("catboost", "ridge"),
) -> Tuple[str, Dict[str, float]]:
    """Pick a model class for a cluster by inner CV with Occam tie-break."""
    member_ids = list(np.where(cluster_mask)[0])
    cluster_X = X_train_full[member_ids]
    cv_scores: Dict[str, float] = {}
    for model_type in candidate_pool:
        cv_scores[model_type] = _inner_cv_mae(model_type, cluster_X, member_ids, start_date, horizon)

    finite = {k: v for k, v in cv_scores.items() if np.isfinite(v)}
    if not finite:
        return "catboost", cv_scores

    best = min(finite, key=finite.get)
    # Occam tie-break: prefer the simpler ridge when it is within tau.
    tau = OCCAM_TAU_SMALL if len(member_ids) < 4 else OCCAM_TAU_LARGE
    if "ridge" in finite and "catboost" in finite:
        if finite["ridge"] <= finite["catboost"] * (1.0 + tau):
            best = "ridge"
    return best, cv_scores


# ---------------------------------------------------------------------------
# Forecast generation: clustered-local and reference branches
# ---------------------------------------------------------------------------


def forecast_global(
    model_type: str,
    X_train: np.ndarray,
    start_date: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Single multi-segment model trained jointly across the panel."""
    return _forecast_with(model_type, X_train, list(range(X_train.shape[0])), start_date, horizon)


def forecast_per_series_local(
    model_type: str,
    X_train: np.ndarray,
    start_date: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """One model per series.  Reuses the multi-segment harness for
    feature engineering parity, but each model sees only one segment."""
    out: Dict[str, np.ndarray] = {}
    for sid in range(X_train.shape[0]):
        local = X_train[[sid]]
        try:
            fc = _forecast_with(model_type, local, [sid], start_date, horizon)
            out.update(fc)
        except Exception:
            continue
    return out


def forecast_clustered_local(
    X_train: np.ndarray,
    cluster_labels: np.ndarray,
    model_mapping: Dict[int, str],
    start_date: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Per-cluster training using the model class chosen by `select_model_for_cluster`."""
    out: Dict[str, np.ndarray] = {}
    for cid in np.unique(cluster_labels):
        ids = list(np.where(cluster_labels == cid)[0])
        model_type = model_mapping.get(int(cid), "catboost")
        try:
            fc = _forecast_with(model_type, X_train[ids], ids, start_date, horizon)
            out.update(fc)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Closed-form baselines (not via ETNA)
# ---------------------------------------------------------------------------


def forecast_naive_seasonal(
    X_train: np.ndarray,
    horizon: int,
    seasonality: int = 7,
) -> Dict[str, np.ndarray]:
    """Repeat the last seasonal cycle as the forecast."""
    out: Dict[str, np.ndarray] = {}
    for sid in range(X_train.shape[0]):
        history = X_train[sid]
        if len(history) < seasonality:
            continue
        last_cycle = history[-seasonality:]
        reps = int(np.ceil(horizon / seasonality))
        forecast = np.tile(last_cycle, reps)[:horizon]
        out[f"region_{sid}"] = forecast
    return out


def forecast_ets(
    X_train: np.ndarray,
    horizon: int,
    seasonality: int = 7,
) -> Dict[str, np.ndarray]:
    """ETS per series via statsmodels.  Falls back to naive on convergence failure."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    out: Dict[str, np.ndarray] = {}
    naive = forecast_naive_seasonal(X_train, horizon, seasonality)
    for sid in range(X_train.shape[0]):
        history = X_train[sid]
        if len(history) < 2 * seasonality + 10:
            out[f"region_{sid}"] = naive[f"region_{sid}"]
            continue
        try:
            model = ExponentialSmoothing(
                history,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonality,
                initialization_method="estimated",
            ).fit()
            forecast = np.asarray(model.forecast(horizon), dtype=float)
            if np.isfinite(forecast).all():
                out[f"region_{sid}"] = forecast
            else:
                out[f"region_{sid}"] = naive[f"region_{sid}"]
        except Exception:
            out[f"region_{sid}"] = naive[f"region_{sid}"]
    return out


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


@dataclass
class MethodSpec:
    name: str
    fn: Callable
    needs_clustering: bool = False
    description: str = ""


def build_method_registry(
    cluster_labels_provider: Optional[Callable[[np.ndarray, str], Tuple[np.ndarray, Dict[int, str]]]] = None,
):
    """Return the canonical comparison set for the main results table.

    The clustered-local entry is parameterized by ``cluster_labels_provider``
    so the orchestration layer can supply the partition and selection
    decisions computed once per (dataset, horizon).
    """
    methods: List[MethodSpec] = [
        MethodSpec(
            name="naive_seasonal_7",
            fn=lambda X_train, start_date, horizon: forecast_naive_seasonal(X_train, horizon, seasonality=7),
            description="Last-week repetition",
        ),
        MethodSpec(
            name="ets",
            fn=lambda X_train, start_date, horizon: forecast_ets(X_train, horizon, seasonality=7),
            description="Holt-Winters additive ETS, weekly seasonality",
        ),
        MethodSpec(
            name="local_catboost",
            fn=lambda X_train, start_date, horizon: forecast_per_series_local("catboost", X_train, start_date, horizon),
            description="One CatBoost model per series",
        ),
        MethodSpec(
            name="global_catboost",
            fn=lambda X_train, start_date, horizon: forecast_global("catboost", X_train, start_date, horizon),
            description="Single CatBoost across panel",
        ),
        MethodSpec(
            name="global_ridge",
            fn=lambda X_train, start_date, horizon: forecast_global("ridge", X_train, start_date, horizon),
            description="Single Ridge across panel",
        ),
        MethodSpec(
            name="global_catboost_id",
            fn=lambda X_train, start_date, horizon: forecast_global("catboost_id", X_train, start_date, horizon),
            description="Single CatBoost across panel with native categorical series-ID feature",
        ),
        MethodSpec(
            name="global_ridge_id",
            fn=lambda X_train, start_date, horizon: forecast_global("ridge_id", X_train, start_date, horizon),
            description="Single Ridge across panel with per-segment mean encoding",
        ),
    ]
    if cluster_labels_provider is not None:
        def _cl_occam(X_train, start_date, horizon):
            labels, mapping = cluster_labels_provider(X_train, start_date)
            return forecast_clustered_local(X_train, labels, mapping, start_date, horizon)

        methods.append(
            MethodSpec(
                name="cl_occam",
                fn=_cl_occam,
                needs_clustering=True,
                description="DTW-clustered + per-cluster CV-Occam selection",
            )
        )
    return methods
