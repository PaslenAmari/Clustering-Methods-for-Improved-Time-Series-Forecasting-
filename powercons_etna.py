"""
Global Electricity Load Clustering and Forecasting
- Dataset: Worldwide Electricity Load Dataset
- Objective: Compare the predictive performance of a unified global model versus 
  a cluster-ensemble approach on a highly heterogeneous electricity load dataset.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import subprocess
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as scipy_stats

print("\n" + "=" * 80)
print("CHECKING IMPORTS")
print("=" * 80)

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
try:
    from dtaidistance.dtw import distance as dtw_distance
    DTW_AVAILABLE = True
    print("[+] dtaidistance")
except Exception:
    DTW_AVAILABLE = False
    print("[-] dtaidistance missing")

SKTIME_VENV_PATH = os.environ.get("SKTIME_VENV_PATH", "/opt/sktime_env/bin/python")
if not os.path.exists(SKTIME_VENV_PATH):
    SKTIME_VENV_PATH = sys.executable

try:
    res = subprocess.run(
        [SKTIME_VENV_PATH, "-c", "import sktime; print('OK')"],
        capture_output=True, text=True, timeout=15
    )
    SKTIME_AVAILABLE = "OK" in res.stdout
    print(f"[+] sktime (via {SKTIME_VENV_PATH})" if SKTIME_AVAILABLE else "[-] sktime check failed")
except Exception:
    SKTIME_AVAILABLE = False
    print("[-] sktime check failed")

AVAILABLE_MODELS = []
try:
    from etna.models import CatBoostMultiSegmentModel, SklearnMultiSegmentModel
    from etna.pipeline import Pipeline
    from etna.datasets import TSDataset
    from etna.transforms import (
        LagTransform,
        DateFlagsTransform,
        StandardScalerTransform,
        TimeSeriesImputerTransform
    )

    AVAILABLE_MODELS.extend(["catboost", "linear"])
    print("[+] ETNA + Core Models (CatBoost, Linear)")
except Exception as e:
    print(f"[-] ETNA core missing: {e}")

try:
    from etna.models import ProphetModel
    AVAILABLE_MODELS.append("prophet")
    print("[+] ETNA + ProphetModel")
except Exception:
    print("[-] ETNA ProphetModel missing")

try:
    from etna.models import SARIMAXModel
    AVAILABLE_MODELS.append("sarimax")
    print("[+] ETNA + SARIMAXModel")
except Exception:
    print("[-] ETNA SARIMAXModel missing")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

FREQ = "D"
LAGS = [1, 2, 3, 7, 14, 30]

# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def detect_datetime_col(df: pd.DataFrame):
    """Identifies the datetime column based on standard naming conventions."""
    candidates = []
    for col in df.columns:
        c = str(col).lower()
        if "time" in c or "date" in c or "datetime" in c or "timestamp" in c:
            candidates.append(col)
    return candidates[0] if candidates else df.columns[0]

def detect_target_col(df: pd.DataFrame, dt_col):
    """Identifies the target variable column (electricity load/demand)."""
    for col in df.columns:
        if col == dt_col:
            continue
        c = str(col).lower()
        if any(x in c for x in ["load", "mw", "demand", "value", "consumption"]):
            return col

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != dt_col]
    return numeric_cols[0] if numeric_cols else None

def parse_single_csv(path):
    """Parses a single region's CSV file and resamples it to a daily frequency."""
    df = pd.read_csv(path, low_memory=False)

    if df.empty or df.shape[1] < 2:
        return None

    dt_col = detect_datetime_col(df)
    target_col = detect_target_col(df, dt_col)

    if target_col is None:
        return None

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")

    df = df.dropna(subset=[dt_col, target_col])
    if df.empty:
        return None

    df = df[[dt_col, target_col]].copy()
    df = df.set_index(dt_col).sort_index()

    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    # Resample to daily mean to standardize the variance across different regions
    daily = df[target_col].resample("D").mean()
    daily = daily.where(daily > 0, np.nan)

    if daily.notna().sum() < 100:
        return None

    return daily

def get_electricity_data():
    """Compiles valid regional datasets into a unified aligned dataframe."""
    print("\nPreparing Worldwide Electricity Load Dataset...")

    base_path = os.path.join(
        "data",
        "Worldwide_electricity_load",
        "Worldwide Electricity Load Dataset",
        "GloElecLoad"
    )

    if not os.path.exists(base_path):
        print(f"  [!] Path not found: {base_path}")
        return None, None, None

    all_series = {}

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".csv") or "Additional_Information" in file or file.startswith("."):
                continue

            path = os.path.join(root, file)
            region_name = os.path.basename(root)
            if region_name == "GloElecLoad":
                region_name = file.replace(".csv", "")

            try:
                series = parse_single_csv(path)
                if series is not None:
                    key = region_name.strip()
                    if key in all_series:
                        key = f"{key}_{file.replace('.csv','')}"
                    all_series[key] = series
            except Exception:
                pass

    if not all_series:
        return None, None, None

    combined = pd.DataFrame(all_series)

    # Establish a fixed chronological window for cross-sectional alignment
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-12-31")
    idx = pd.date_range(start_date, end_date, freq="D")
    combined = combined.reindex(idx)

    # Filter regions by data completeness threshold
    coverage = combined.notna().mean().sort_values(ascending=False)
    valid_cols = coverage[coverage >= 0.70].index.tolist()

    if len(valid_cols) < 8:
        valid_cols = coverage.head(min(30, len(coverage))).index.tolist()

    combined = combined[valid_cols]
    combined = combined.loc[:, combined.notna().sum() >= 200]

    if combined.shape[1] < 4:
        return None, None, None

    # Handle missing values through linear interpolation
    combined = combined.interpolate(method="linear", limit_direction="both")
    combined = combined.ffill().bfill()

    # Eliminate regions with zero variance
    stds = combined.std(axis=0)
    combined = combined.loc[:, stds > 1e-8]

    X = combined.T.values.astype(float)
    region_names = combined.columns.tolist()

    print(f"  Extracted {X.shape[0]} regions over {X.shape[1]} days.")
    return X, region_names, start_date.strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------
# CLUSTERING METHODOLOGY
# -----------------------------------------------------------------------------
def cluster_dtw(X, n_clusters=4):
    """
    Performs time series clustering using Dynamic Time Warping (DTW).
    Features are standardized to cluster based on temporal shape patterns 
    rather than absolute magnitude scale.
    """
    print("\n" + "=" * 80)
    print("APPROACH: DTW Distance + AgglomerativeClustering")
    print("=" * 80)

    n = X.shape[0]
    if n < 3 or not DTW_AVAILABLE:
        return np.zeros(n, dtype=int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X.T).T

    D = np.zeros((n, n))
    print(f"  Calculating DTW distance matrix ({n}×{n})...")
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_distance(X_sc[i], X_sc[j])

    n_clusters = min(n_clusters, max(2, n // 4))

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    ).fit_predict(D)

    sizes = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  Cluster sizes: {sizes}")

    # Fallback to KMeans if Agglomerative Clustering produces a single dominating cluster
    counts = np.bincount(labels)
    if counts.max() / counts.sum() > 0.85 and n >= 4:
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(D)
        sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"  New Cluster sizes (KMeans): {sizes}")

    return labels

# -----------------------------------------------------------------------------
# ETNA FRAMEWORK UTILITIES
# -----------------------------------------------------------------------------
def make_ts_dataset(X_sub, segment_ids, start_date, freq=FREQ):
    """Converts numpy arrays into ETNA TSDataset format."""
    timestamps = pd.date_range(start_date, periods=X_sub.shape[1], freq=freq)
    rows = []
    for i, sid in enumerate(segment_ids):
        rows.append(pd.DataFrame({
            "timestamp": timestamps,
            "segment": f"region_{sid}",
            "target": X_sub[i]
        }))
    return TSDataset(TSDataset.to_dataset(pd.concat(rows, ignore_index=True)), freq=freq)

def build_transforms(with_imputer=False):
    """Constructs the feature engineering pipeline for ETNA models."""
    transforms = []
    if with_imputer:
        transforms.append(TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"))
    transforms.extend([
        LagTransform(in_column="target", lags=LAGS),
        DateFlagsTransform(day_number_in_week=True, is_weekend=True),
        StandardScalerTransform(in_column="target")
    ])
    return transforms

def get_model(model_type):
    """Instantiates the specified machine learning or statistical model."""
    if model_type == "linear":
        return SklearnMultiSegmentModel(
            regressor=make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0),
                Ridge(alpha=0.5) 
            )
        )
    elif model_type == "prophet":
        return ProphetModel()
    elif model_type == "sarimax":
        return SARIMAXModel(order=(1, 1, 1), disp=False)
    else:
        return CatBoostMultiSegmentModel(
            iterations=250, depth=4, learning_rate=0.05, random_seed=42, logging_level="Silent"
        )

# -----------------------------------------------------------------------------
# MODEL SELECTION AND VALIDATION
# -----------------------------------------------------------------------------
def select_model_for_cluster_via_cv(X_train, cluster_mask, horizon, cluster_id, start_date):
    """
    Evaluates candidate models for a specific cluster using internal cross-validation.
    Implements Occam's Razor principle to prefer simpler models when performance is comparable.
    """
    cluster_ids = np.where(cluster_mask)[0]
    cv_ids = cluster_ids[:min(len(cluster_ids), 10)]

    print(f"  Cluster {cluster_id}: {len(cluster_ids)} regions")
    
    if len(cv_ids) == 0:
        return "catboost"

    X_cv_tr = X_train[cv_ids, :-horizon]
    X_cv_val = X_train[cv_ids, -horizon:]

    def cv_mae(model_type):
        try:
            ts = make_ts_dataset(X_cv_tr, cv_ids, start_date)
            use_imputer = model_type in ["linear", "prophet", "sarimax"]
            pipe = Pipeline(
                model=get_model(model_type),
                transforms=build_transforms(with_imputer=use_imputer),
                horizon=horizon
            )
            pipe.fit(ts)
            fdf = pipe.forecast().to_pandas()

            maes = []
            for sid in cv_ids:
                seg = f"region_{sid}"
                if seg in fdf.columns.get_level_values(0):
                    fc = fdf[seg]["target"].values
                    if len(fc) == horizon and np.isfinite(fc).all():
                        local_idx = list(cv_ids).index(sid)
                        maes.append(mean_absolute_error(X_cv_val[local_idx], fc))

            return float(np.mean(maes)) if maes else np.inf
        except Exception:
            return np.inf

    candidate_models = ["catboost", "linear"]
    results = {m: cv_mae(m) for m in candidate_models}
    
    res_str = " | ".join([f"{k.capitalize()}: {v:.1f}" if np.isfinite(v) else f"{k.capitalize()}: inf" for k, v in results.items()])
    print(f"      CV MAE — {res_str}")

    best_model = min(results, key=results.get)
    
    # Occam's Razor Heuristic:
    # Linear models are more robust against overfitting on small datasets.
    # We penalize complex models and force selection of the linear model 
    # if the performance degradation is within an acceptable margin.
    if "linear" in results and "catboost" in results:
        if results["linear"] <= results["catboost"] * 1.10:
            best_model = "linear"
        elif len(cluster_ids) <= 3 and results["linear"] <= results["catboost"] * 1.25:
            best_model = "linear"

    if not np.isfinite(results.get(best_model, np.inf)):
        best_model = "catboost"

    print(f"    → Selected: {best_model.upper()}")
    return best_model

# -----------------------------------------------------------------------------
# FORECASTING PIPELINES
# -----------------------------------------------------------------------------
def build_global_model(X_train, horizon, start_date):
    """Trains a single, unified CatBoost model across all geographical regions."""
    print("GLOBAL MODEL: Building CatBoostMultiSegmentModel ...")
    ts = make_ts_dataset(X_train, list(range(X_train.shape[0])), start_date)

    pipe = Pipeline(
        model=CatBoostMultiSegmentModel(
            iterations=300, depth=5, learning_rate=0.03, random_seed=42, logging_level="Silent"
        ),
        transforms=build_transforms(with_imputer=False),
        horizon=horizon
    )
    pipe.fit(ts)
    fdf = pipe.forecast().to_pandas()
    return {seg: fdf[seg]["target"].values for seg in fdf.columns.get_level_values(0).unique()}

def build_cluster_models(X_train, cluster_labels, horizon, model_mapping, start_date):
    """Trains specialized models for each respective DTW-identified cluster."""
    print("CLUSTER MODELS: Building per-cluster models with diversity ...")
    cluster_forecasts = {}

    for cid in np.unique(cluster_labels):
        ids = np.where(cluster_labels == cid)[0]
        choice = model_mapping.get(cid, "catboost")

        print(f"  Cluster {cid} ({len(ids)} regions) → {choice.upper()}")
        try:
            ts = make_ts_dataset(X_train[ids], list(ids), start_date)
            pipe = Pipeline(
                model=get_model(choice),
                transforms=build_transforms(with_imputer=choice in ["linear", "prophet", "sarimax"]),
                horizon=horizon
            )
            pipe.fit(ts)
            fdf = pipe.forecast().to_pandas()

            local_forecasts = {}
            for seg in fdf.columns.get_level_values(0).unique():
                local_forecasts[seg] = fdf[seg]["target"].values
            cluster_forecasts[cid] = local_forecasts
        except Exception:
            pass

    return cluster_forecasts

def aggregate_cluster_forecasts(cluster_labels, cluster_forecasts):
    """Aggregates cluster-specific predictions into a unified ensemble forecast."""
    ensemble = {}
    for idx, cid in enumerate(cluster_labels):
        seg = f"region_{idx}"
        if cid in cluster_forecasts and seg in cluster_forecasts[cid]:
            ensemble[seg] = cluster_forecasts[cid][seg]
    return ensemble

# -----------------------------------------------------------------------------
# EVALUATION & STATISTICAL TESTING
# -----------------------------------------------------------------------------
def evaluate_window(global_f, ensemble_f, Xval, horizon):
    """
    Computes regression metrics and extracts regional errors 
    for subsequent non-parametric statistical testing.
    """
    g_abs, e_abs = [], []
    g_smape, e_smape = [], []
    
    region_g_mae = []
    region_e_mae = []

    for seg_key, g_fc in global_f.items():
        mid = int(seg_key.replace("region_", ""))
        actual = Xval[mid, -horizon:]
        e_fc = ensemble_f.get(seg_key, g_fc)

        if not (np.isfinite(g_fc).all() and np.isfinite(e_fc).all()):
            continue

        reg_g_mae = mean_absolute_error(actual, g_fc)
        reg_e_mae = mean_absolute_error(actual, e_fc)
        
        region_g_mae.append(reg_g_mae)
        region_e_mae.append(reg_e_mae)

        g_abs.append(reg_g_mae)
        e_abs.append(reg_e_mae)

        g_den = (np.abs(actual) + np.abs(g_fc)) / 2 + 1e-8
        e_den = (np.abs(actual) + np.abs(e_fc)) / 2 + 1e-8
        g_smape.append(np.mean(np.abs(actual - g_fc) / g_den) * 100)
        e_smape.append(np.mean(np.abs(actual - e_fc) / e_den) * 100)

    print("GLOBAL Model")
    print(f"  MAE   = {np.mean(g_abs):.2f} | sMAPE = {np.mean(g_smape):.2f}")
    print("ENSEMBLE Model")
    print(f"  MAE   = {np.mean(e_abs):.2f} | sMAPE = {np.mean(e_smape):.2f}")

    return region_g_mae, region_e_mae

# -----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PIPELINE: Global Electricity Load Forecasting")
    print("=" * 80)

    X, region_names, start_date_str = get_electricity_data()
    if X is None or X.shape[0] < 4:
        return

    # Define prediction horizon and chronological backtest windows
    horizon = 30
    n_windows = 4 
    n_test = horizon * n_windows 
    
    test_windows = [
        (X.shape[1] - n_test + i * horizon, X.shape[1] - n_test + (i + 1) * horizon)
        for i in range(n_windows)
    ]

    # Dynamically determine the number of clusters based on dataset volume
    n_clusters = min(6, max(3, X.shape[0] // 5))
    labels_dtw = cluster_dtw(X, n_clusters=n_clusters)

    print("\n" + "=" * 60)
    print("MODEL SELECTION for DTW (CV Competition)")
    print("=" * 60)
    
    mapping_dtw = {
        cid: select_model_for_cluster_via_cv(X, labels_dtw == cid, horizon, cid, start_date_str)
        for cid in np.unique(labels_dtw)
    }

    print(f"  Final mapping: {mapping_dtw}")

    all_reg_g_mae, all_reg_e_mae = [], []

    # Execute temporal cross-validation
    for i, (train_end, test_end) in enumerate(test_windows):
        print(f"\n--- Backtest Window {i+1}/{n_windows} ---")
        Xtrain = X[:, :train_end]
        Xval = X[:, train_end:test_end]

        global_f = build_global_model(Xtrain, horizon, start_date_str)
        cluster_fc = build_cluster_models(Xtrain, labels_dtw, horizon, mapping_dtw, start_date_str)
        ensemble_f = aggregate_cluster_forecasts(labels_dtw, cluster_fc)

        reg_g_mae, reg_e_mae = evaluate_window(global_f, ensemble_f, Xval, horizon)
        all_reg_g_mae.extend(reg_g_mae)
        all_reg_e_mae.extend(reg_e_mae)

    # Apply Wilcoxon Signed-Rank Test for robust paired-sample evaluation
    # This non-parametric test is optimal for regional error distributions 
    # that violate normality assumptions.
    w_stat, w_pval = scipy_stats.wilcoxon(all_reg_g_mae, all_reg_e_mae, alternative='greater')

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Wilcoxon Signed-Rank Test: statistic = {w_stat:.1f}, p-value = {w_pval:.4f}")

    # Interpret statistical significance at the alpha = 0.05 level
    if np.isfinite(w_pval) and w_pval < 0.05:
        print("→ ENSEMBLE is significantly BETTER ✓")
    else:
        print("→ Difference NOT significant")
        
    print("✓ Pipeline complete.")

if __name__ == "__main__":
    main()
