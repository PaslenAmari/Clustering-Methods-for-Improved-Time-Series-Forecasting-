"""
The Project: Oil & Gas Well Production Clustering and Forecasting
- Dataset: Volve Field (Equinor, Norway) - Real Daily Production Data (.xlsx)
- Clustering daily oil production time series (wells)
- Applying Global vs Cluster-Ensemble approach with diverse ETNA models
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import subprocess
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

print("\n" + "=" * 80)
print("CHECKING IMPORTS")
print("=" * 80)

# ----------------------------------------------------------------------------
# 1. DEPENDENCIES CHECK
# ----------------------------------------------------------------------------
try:
    from dtaidistance.dtw import distance as dtw_distance
    DTW_AVAILABLE = True
    print("[+] dtaidistance")
except ImportError:
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

# ----------------------------------------------------------------------------
# 2. ETNA & MODELS CHECK
# ----------------------------------------------------------------------------
AVAILABLE_MODELS = []
try:
    from etna.models import CatBoostMultiSegmentModel, SklearnMultiSegmentModel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from etna.pipeline import Pipeline
    from etna.datasets import TSDataset
    from etna.transforms import LagTransform, DateFlagsTransform, StandardScalerTransform, TimeSeriesImputerTransform
    
    AVAILABLE_MODELS.extend(["catboost", "linear"])
    print("[+] ETNA + Core Models (CatBoost, Linear)")
except Exception as e:
    print(f"[-] ETNA core missing: {e}")

try:
    from etna.models import ProphetModel
    AVAILABLE_MODELS.append("prophet")
    print("[+] ETNA + ProphetModel")
except ImportError:
    print("[-] ETNA ProphetModel missing")

try:
    from etna.models import SARIMAXModel
    AVAILABLE_MODELS.append("sarimax")
    print("[+] ETNA + SARIMAXModel")
except ImportError:
    print("[-] ETNA SARIMAXModel missing")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
DATA_DIR    = "data"
VOLVE_DIR   = os.path.join(DATA_DIR, "VOLVE_PRODUCTION_DATA")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VOLVE_DIR,   exist_ok=True)

FREQ = "D"
LAGS = [1, 2, 3, 7, 14, 30]

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace("\xa0", "").str.replace(" ", "").str.replace(",", "."),
        errors="coerce"
    )

def get_volve_data():
    print("\nPreparing Volve Field Production Dataset...")
    excel_path = os.path.join(VOLVE_DIR, "Volve production data.xlsx")

    if not os.path.exists(excel_path):
        print(f"  [!] File not found at: {excel_path}")
        return None, None

    print(f"  Loading dataset from {excel_path} (may take 30-60s for Excel)...")
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
    except Exception as e:
        print(f"  [!] Failed to read Excel: {e}")
        return None, None

    df.columns = [str(c).strip().upper() for c in df.columns]
    required = ["DATEPRD", "NPD_WELL_BORE_NAME", "BORE_OIL_VOL"]
    if not all(c in df.columns for c in required):
        print(f"  [!] Missing columns. Found: {df.columns.tolist()}")
        return None, None

    extra = [c for c in ["FLOW_KIND", "WELL_TYPE"] if c in df.columns]
    df = df[required + extra].copy()

    df["DATEPRD"]      = pd.to_datetime(df["DATEPRD"], errors="coerce", dayfirst=True)
    df["BORE_OIL_VOL"] = parse_numeric_series(df["BORE_OIL_VOL"])

    if "FLOW_KIND" in df.columns:
        df = df[df["FLOW_KIND"].astype(str).str.strip().str.lower() == "production"]
    if "WELL_TYPE" in df.columns:
        df = df[df["WELL_TYPE"].astype(str).str.strip().str.upper() != "wi"]

    df = df.dropna(subset=["DATEPRD", "NPD_WELL_BORE_NAME", "BORE_OIL_VOL"])
    df = df[df["BORE_OIL_VOL"] >= 0]

    df_pivot = df.pivot_table(index="NPD_WELL_BORE_NAME", columns="DATEPRD", values="BORE_OIL_VOL", aggfunc="sum").sort_index(axis=1)

    start_date = pd.to_datetime("2013-01-01")
    end_date   = pd.to_datetime("2014-12-31")
    mask       = (df_pivot.columns >= start_date) & (df_pivot.columns <= end_date)
    df_f       = df_pivot.loc[:, mask]

    if df_f.shape[1] == 0: df_f = df_pivot.copy()

    df_f = df_f.dropna(thresh=max(int(0.5 * df_f.shape[1]), 1))
    df_f = df_f.fillna(0.0)

    # Убираем "мертвые" скважины
    active = (df_f > 0).sum(axis=1) / df_f.shape[1]
    df_f   = df_f.loc[active >= 0.05]

    X          = df_f.values.astype(float)
    well_names = df_f.index.tolist()

    print(f"  Extracted {X.shape[0]} well time-series over {X.shape[1]} days.")
    print(f"  Wells: {well_names}")
    return X, well_names

# ============================================================================
# CLUSTERING
# ============================================================================

def is_degenerate_clustering(labels, max_share=0.95):
    counts = np.bincount(labels)
    return len(counts) < 2 or counts.max() / counts.sum() > max_share

def cluster_dtw(X, n_clusters=2):
    if not DTW_AVAILABLE: return np.zeros(X.shape[0], dtype=int)
    print("\n" + "=" * 80 + "\nAPPROACH: DTW Distance + AgglomerativeClustering\n" + "=" * 80)
    
    n = X.shape[0]
    if n < 3: return np.zeros(n, dtype=int)

    X_sc = np.array([(x - x.mean()) / (x.std() + 1e-8) for x in X])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(X_sc[i], X_sc[j])
            D[i, j] = D[j, i] = d

    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average").fit_predict(D)
    
    if len(np.unique(labels)) > 1:
        print(f"  Silhouette Score:       {silhouette_score(D, labels, metric='precomputed'):.4f}")
    
    if is_degenerate_clustering(labels): return np.zeros(n, dtype=int)
    return labels

# ============================================================================
# ETNA UTILS & MODEL SELECTION
# ============================================================================

def make_ts_dataset(X_sub, meter_ids, freq=FREQ):
    ts = pd.date_range("2013-01-01", periods=X_sub.shape[1], freq=freq)
    rows = [pd.DataFrame({"timestamp": ts, "segment": f"well_{mid}", "target": X_sub[i]}) for i, mid in enumerate(meter_ids)]
    return TSDataset(TSDataset.to_dataset(pd.concat(rows, ignore_index=True)), freq=freq)

def build_transforms(include_imputer=False):
    """
    Добавляем Imputer для статистических моделей, чтобы заполнить NaNs и нули.
    CatBoost справится и без него, но Prophet/SARIMAX упадут без импутации.
    """
    transforms = []
    if include_imputer:
        transforms.append(TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"))
    
    transforms.extend([
        LagTransform(in_column="target", lags=LAGS),
        DateFlagsTransform(day_number_in_week=True, is_weekend=True),
        StandardScalerTransform(in_column="target")
    ])
    return transforms

def get_etna_model_by_name(model_type):
    if model_type == "linear":
        return SklearnMultiSegmentModel(regressor=make_pipeline(SimpleImputer(strategy="constant", fill_value=0), Ridge()))
    elif model_type == "prophet":
        from etna.models import ProphetModel
        return ProphetModel()
    elif model_type == "sarimax":
        from etna.models import SARIMAXModel
        # Отключаем вывод (disp=False) чтобы не спамить в консоль L-BFGS-B кодом
        return SARIMAXModel(order=(1, 1, 1), disp=False)
    else:
        return CatBoostMultiSegmentModel(iterations=100, depth=3, learning_rate=0.05, random_seed=42, logging_level="Silent")

def select_model_for_cluster_via_cv(X_train, cluster_mask, horizon, cluster_id):
    cluster_ids = np.where(cluster_mask)[0]
    cv_ids      = cluster_ids[:min(len(cluster_ids), 15)]

    print(f"  Cluster {cluster_id}: {len(cluster_ids)} wells")
    print(f"    Running internal CV (horizon={horizon}, {len(cv_ids)} wells)...")

    X_cv_tr  = X_train[cv_ids, :-horizon]
    X_cv_val = X_train[cv_ids, -horizon:]
    ts_train = pd.date_range("2013-01-01", periods=X_cv_tr.shape[1], freq=FREQ)

    def cv_mae(model_type):
        try:
            # Скрываем предупреждения от statsmodels
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                rows = [pd.DataFrame({"timestamp": ts_train, "segment": f"well_{mid}", "target": X_cv_tr[i]}) for i, mid in enumerate(cv_ids)]
                ts = TSDataset(TSDataset.to_dataset(pd.concat(rows, ignore_index=True)), freq=FREQ)
                
                mdl = get_etna_model_by_name(model_type)
                # Prophet и SARIMAX требуют импутацию пропусков и нулей
                needs_imputer = model_type in ["prophet", "sarimax"]
                pipe = Pipeline(model=mdl, transforms=build_transforms(include_imputer=needs_imputer), horizon=horizon)
                
                pipe.fit(ts)
                fdf = pipe.forecast().to_pandas()
                
                maes = []
                for seg in fdf.columns.get_level_values(0).unique():
                    ms = str(seg).replace("well_", "")
                    if ms.isdigit() and int(ms) in cv_ids:
                        fc = fdf[seg]["target"].values
                        if len(fc) == horizon and not np.isnan(fc).any():
                            maes.append(mean_absolute_error(X_cv_val[list(cv_ids).index(int(ms))], fc))
                return np.mean(maes) if maes else np.inf
        except Exception:
            return np.inf

    results = {m: cv_mae(m) for m in AVAILABLE_MODELS}
    res_str = " | ".join([f"{k.capitalize()}: {v:.1f}" if np.isfinite(v) else f"{k.capitalize()}: inf" for k, v in results.items()])
    print(f"      CV MAE — {res_str}")

    best_model = min(results, key=results.get)
    choice = best_model if np.isfinite(results[best_model]) else "catboost"
    print(f"    → Selected: {choice.upper()}")
    return choice

# ============================================================================
# GLOBAL & CLUSTER MODELS
# ============================================================================

def build_global_model(X_train, horizon):
    print("GLOBAL MODEL: Building CatBoostMultiSegmentModel ...")
    ts = make_ts_dataset(X_train, list(range(X_train.shape[0])))
    pipe = Pipeline(
        model=CatBoostMultiSegmentModel(iterations=300, depth=5, learning_rate=0.03, random_seed=42, logging_level="Silent"),
        transforms=build_transforms(include_imputer=False), horizon=horizon
    )
    pipe.fit(ts)
    fdf = pipe.forecast().to_pandas()
    segs = fdf.columns.get_level_values(0).unique()
    return {seg: fdf[seg]["target"].values for seg in segs if len(fdf[seg]["target"].values) == horizon}

def build_cluster_models(X_train, cluster_labels, horizon, model_mapping):
    print("CLUSTER MODELS: Building per-cluster models with diversity ...")
    all_fc = {}
    for cid in np.unique(cluster_labels):
        ids = np.where(cluster_labels == cid)[0]
        choice = model_mapping.get(cid, "catboost")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if choice == "catboost":
                mdl = CatBoostMultiSegmentModel(iterations=300, depth=4, learning_rate=0.05, random_seed=42, logging_level="Silent")
            else:
                mdl = get_etna_model_by_name(choice)

            print(f"  Cluster {cid} ({len(ids)} wells) → {choice.upper()}")
            
            ts = make_ts_dataset(X_train, list(ids))
            needs_imputer = choice in ["prophet", "sarimax"]
            pipe = Pipeline(model=mdl, transforms=build_transforms(include_imputer=needs_imputer), horizon=horizon)
            
            try:
                pipe.fit(ts)
                fdf = pipe.forecast().to_pandas()
                all_fc[cid] = {seg: fdf[seg]["target"].values for seg in fdf.columns.get_level_values(0).unique() if len(fdf[seg]["target"].values) == horizon}
            except Exception as e:
                print(f"  [!] Model failed for cluster {cid}")

    return all_fc

def aggregate_cluster_forecasts(cluster_labels, cluster_forecasts):
    ensemble = {}
    for mid in range(len(cluster_labels)):
        cid, seg_key = cluster_labels[mid], f"well_{mid}"
        if cid in cluster_forecasts and seg_key in cluster_forecasts[cid]:
            ensemble[seg_key] = cluster_forecasts[cid][seg_key]
    return ensemble

# ============================================================================
# EVALUATION & MAIN
# ============================================================================

def evaluate_window(global_f, ensemble_f, Xval, horizon):
    g_errs, e_errs, g_abs, e_abs = [], [], [], []
    for seg_key, g_fc in global_f.items():
        mid = int(seg_key.replace("well_", ""))
        actual = Xval[mid, -horizon:]
        e_fc = ensemble_f.get(seg_key, g_fc)

        if np.isnan(g_fc).any() or np.isnan(e_fc).any(): continue

        g_errs.extend((actual - g_fc).tolist()); e_errs.extend((actual - e_fc).tolist())
        g_abs.append(mean_absolute_error(actual, g_fc)); e_abs.append(mean_absolute_error(actual, e_fc))

    print(f"GLOBAL Model MAE:   {np.mean(g_abs):.2f}")
    print(f"ENSEMBLE Model MAE: {np.mean(e_abs):.2f}")
    return g_errs, e_errs

def main():
    print("=" * 80 + "\nPIPELINE: Real O&G Well Production Forecasting (Volve)\n" + "=" * 80)
    X, well_names = get_volve_data()
    if X is None: return

    horizon, n_windows = 30, 2
    n_test = max(horizon * 2, int(X.shape[1] * 0.20))
    test_windows = [(X.shape[1] - n_test + i * horizon, X.shape[1] - n_test + (i + 1) * horizon) for i in range(n_windows)]

    labels_dtw = cluster_dtw(X, n_clusters=2 if X.shape[0] < 10 else 3)

    print("\n" + "=" * 60 + "\nMODEL SELECTION for DTW (CV Competition)\n" + "=" * 60)
    mapping_dtw = {cid: select_model_for_cluster_via_cv(X, labels_dtw == cid, horizon, cid) for cid in np.unique(labels_dtw)}

    all_g_err, all_e_err = [], []
    for i, (train_end, test_end) in enumerate(test_windows):
        print(f"\n--- Backtest Window {i+1}/{n_windows} ---")
        Xtrain, Xval = X[:, :train_end], X[:, train_end:test_end]
        global_f = build_global_model(Xtrain, horizon)
        ensemble_f = aggregate_cluster_forecasts(labels_dtw, build_cluster_models(Xtrain, labels_dtw, horizon, mapping_dtw))
        g_errs, e_errs = evaluate_window(global_f, ensemble_f, Xval, horizon)
        all_g_err.extend(g_errs); all_e_err.extend(e_errs)

    # Simple DM Stat (no print wall)
    d = np.array(all_g_err)**2 - np.array(all_e_err)**2
    var_d = max(np.var(d, ddof=1), 1e-10)
    dm_stat = np.mean(d) / np.sqrt(var_d / len(d))
    pval = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))

    print("\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80)
    print(f"Diebold-Mariano Test: DM = {dm_stat:.4f}, p-value = {pval:.4f}")
    if pval < 0.05:
        print("→ ENSEMBLE is significantly BETTER ✓" if dm_stat > 0 else "→ GLOBAL is significantly BETTER")
    else: print("→ Difference NOT significant")
    print("✓ Pipeline complete.")

if __name__ == "__main__": main()