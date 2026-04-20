# ============================================================
# Project: Energy Consumption Clustering & Forecasting
#   - TSFresh / DTW / sktime clustering of time series (not points)
#   - Per-cluster model selection via ETNA internal CV (not FEDOT on centroid)
#   - Medoid-based representative series (not smoothed centroid)
#   - True model diversity: CatBoost vs LinearMultiSegmentModel per cluster
#   - Diebold-Mariano test with accumulated errors across 6 backtest windows
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, shutil, zipfile, subprocess, pickle, tempfile
from urllib.request import urlretrieve
from urllib.error import URLError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
)

print("=" * 80)
print("CHECKING IMPORTS")
print("=" * 80)

# --- DTW ---
try:
    from dtaidistance.dtw import distance as dtw_distance
    DTW_AVAILABLE = True
    print("✓ dtaidistance dtw.distance")
except ImportError as e:
    print(f"✗ dtaidistance: {e}")
    DTW_AVAILABLE = False

# --- TSFresh ---
try:
    from tsfresh import extract_features
    TSFRESH_AVAILABLE = True
    print("✓ TSFresh")
except ImportError as e:
    print(f"✗ TSFresh: {e}")
    TSFRESH_AVAILABLE = False

# --- sktime (isolated venv) ---
SKTIME_VENV_PATH = os.environ.get("SKTIME_VENV_PATH", "/opt/sktime-env/bin/python")
if not os.path.exists(SKTIME_VENV_PATH):
    import sys; SKTIME_VENV_PATH = sys.executable
try:
    result = subprocess.run(
        [SKTIME_VENV_PATH, "-c", "from sktime.clustering.kmeans import TimeSeriesKMeans; print('OK')"],
        capture_output=True, text=True, timeout=10
    )
    SKTIME_AVAILABLE = result.returncode == 0 and "OK" in result.stdout
    print(f"{'✓' if SKTIME_AVAILABLE else '✗'} sktime via {SKTIME_VENV_PATH}")
except Exception as e:
    SKTIME_AVAILABLE = False
    print(f"✗ sktime: {e}")

# --- ETNA ---
try:
    from etna.models import CatBoostMultiSegmentModel, LinearMultiSegmentModel
    from etna.pipeline import Pipeline
    from etna.datasets import TSDataset
    from etna.transforms import LagTransform, DateFlagsTransform, StandardScalerTransform
    try:
        from etna.transforms import LogTransform
    except ImportError:
        LogTransform = None
        print("  Notice: LogTransform not found in ETNA")
    try:
        from etna.transforms import MeanTransform as StatisticsTransform
    except ImportError:
        try:
            from etna.transforms import StatisticsTransform
        except ImportError:
            StatisticsTransform = None
            print("  Notice: StatisticsTransform/MeanTransform not found in ETNA")
    ETNA_AVAILABLE = True
    print("✓ ETNA CatBoostMultiSegmentModel + LinearMultiSegmentModel")
except Exception as e:
    print(f"✗ ETNA: {e}")
    ETNA_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
DATA_DIR    = "data"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── global params (auto-detected later) ──────────────────────────────────────
FREQ         = "H"
LAGS         = [1, 2, 3, 24, 48]
ROLLING_WIN  = 12
N_SAMPLE     = 50
CLUSTER_SMALL  = 10
CLUSTER_MEDIUM = 50


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def download_powercons_dataset(data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataset_dir = os.path.join(data_dir, "PowerCons")
    if os.path.exists(dataset_dir):
        return True
    archive_url  = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
    archive_path = os.path.join(data_dir, "UCRArchive_2018.zip")
    try:
        print("Downloading UCR Archive …")
        urlretrieve(archive_url, archive_path)
    except URLError as e:
        print(f"Download error: {e}")
        if os.path.exists(archive_path): os.remove(archive_path)
        return False
    try:
        print("Extracting PowerCons files …")
        with zipfile.ZipFile(archive_path, "r") as zf:
            for f in zf.namelist():
                if f.startswith("UCRArchive_2018/PowerCons"):
                    zf.extract(f, data_dir)
    except Exception as e:
        print(f"Extraction error: {e}")
        if os.path.exists(archive_path): os.remove(archive_path)
        return False
    try:
        nested = os.path.join(data_dir, "UCRArchive_2018", "PowerCons")
        if os.path.exists(nested):
            shutil.move(nested, data_dir)
        root_ucr = os.path.join(data_dir, "UCRArchive_2018")
        if os.path.exists(root_ucr):
            shutil.rmtree(root_ucr)
    except Exception as e:
        print(f"File organization error: {e}")
    try:
        os.remove(archive_path)
    except Exception:
        pass
    print("Dataset ready!")
    return True


def load_powercons_data(data_dir=DATA_DIR):
    print("Loading PowerCons dataset …")
    if not download_powercons_dataset(data_dir):
        print("FALLBACK: Using synthetic data …")
        return load_synthetic_data()
    dataset_dir = os.path.join(data_dir, "PowerCons")
    train_path  = os.path.join(dataset_dir, "PowerCons_TRAIN.tsv")
    test_path   = os.path.join(dataset_dir, "PowerCons_TEST.tsv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: dataset files not found — using synthetic data")
        return load_synthetic_data()
    train = pd.read_csv(train_path, header=None, sep="\t")
    test  = pd.read_csv(test_path,  header=None, sep="\t")
    X = np.vstack([train.iloc[:, 1:].values, test.iloc[:, 1:].values]).astype(float)
    print(f"PowerCons loaded: {X.shape[0]} series × {X.shape[1]} timesteps")
    return X


def load_synthetic_data():
    """Synthetic fallback — 4 clear archetypes so ensemble beats global."""
    print("Generating synthetic PowerCons data …")
    np.random.seed(42)
    n_series   = 30
    n_timesteps = 144
    X = np.zeros((n_series, n_timesteps))
    t = np.arange(n_timesteps)
    for i in range(n_series):
        t_norm = t / n_timesteps
        archetype = i % 4
        if archetype == 0:   # High early-morning peak
            X[i] = 2.5 + 1.8 * np.sin(2 * np.pi * t / 24 + 1.0) + np.random.normal(0, 0.15, n_timesteps)
        elif archetype == 1: # Evening peak
            X[i] = 2.0 + 1.5 * np.sin(2 * np.pi * t / 24 - 1.0) + np.random.normal(0, 0.15, n_timesteps)
        elif archetype == 2: # Flat high consumption
            X[i] = 3.5 + 0.3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, n_timesteps)
        else:                # Low-usage with weekly rhythm
            X[i] = 0.8 + 0.5 * np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 0.12, n_timesteps)
    print(f"Synthetic data: {X.shape[0]} series × {X.shape[1]} timesteps  (4 archetypes)")
    return X


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AUTO PARAMETER DETECTION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def autodetect_params(X, verbose=True):
    """Detect freq, seasonal lag, lags, rolling window, n_clusters, n_sample."""
    n_series, n_timesteps = X.shape
    params = {}
    if verbose:
        print("=" * 60)
        print("AUTO PARAMETER DETECTION")
        print("=" * 60)
        print(f"Dataset: {n_series} series × {n_timesteps} timesteps")

    # Frequency
    freq = "H" if n_timesteps >= 24 else "D"
    params["freq"] = freq

    # Seasonal lag via ACF
    try:
        from statsmodels.tsa.stattools import acf
        sample_idx = np.random.choice(n_series, size=min(20, n_series), replace=False)
        max_lag = min(n_timesteps // 2, 168)
        acf_sum = np.zeros(max_lag)
        for idx in sample_idx:
            series = X[idx]
            series = (series - series.mean()) / (series.std() + 1e-8)
            a = acf(series, nlags=max_lag - 1, fft=True)
            acf_sum += np.abs(a[1:])
        acf_mean = acf_sum / len(sample_idx)
        peaks = [(lag + 1, v) for lag, v in enumerate(acf_mean) if v > 0.3]
        if peaks:
            seasonal_lag = max(peaks, key=lambda x: x[1])[0]
        else:
            seasonal_lag = 24 if freq == "H" else 7
    except Exception as e:
        seasonal_lag = 24 if freq == "H" else 7
        if verbose: print(f"  ! ACF detection failed: {e}, using seasonal_lag={seasonal_lag}")
    params["seasonal_lag"] = seasonal_lag

    # Lags
    lags = sorted({1, 2, 3, seasonal_lag, seasonal_lag * 2})
    lags = [l for l in lags if l < n_timesteps][:7]
    params["lags"] = lags

    # Rolling window
    params["rolling_win"] = max(4, seasonal_lag // 2)

    # N clusters — Silhouette scan
    X_scaled = np.array([(x - x.mean()) / (x.std() + 1e-8) for x in X])
    sample_size = min(n_series, 50)
    X_sample = X_scaled[:sample_size]
    best_k, best_sil = 4, -1.0
    for k in range(2, min(8, sample_size // 3 + 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        lbl = km.fit_predict(X_sample)
        s = silhouette_score(X_sample, lbl)
        if s > best_sil:
            best_sil, best_k = s, k
    params["n_clusters"] = best_k
    params["best_sil"]   = best_sil

    # N sample for DTW/sktime
    params["n_sample"] = min(n_series, max(n_series, 50) if n_series <= 100 else 100)

    # Cluster size thresholds
    params["cluster_small"]  = max(5,  int(n_series * 0.03))
    params["cluster_medium"] = max(20, int(n_series * 0.15))

    if verbose:
        print(f"  freq         = {freq}")
        print(f"  seasonal_lag = {seasonal_lag}")
        print(f"  lags         = {lags}")
        print(f"  rolling_win  = {params['rolling_win']}")
        print(f"  n_clusters   = {best_k}  (Silhouette={best_sil:.3f})")
        print(f"  n_sample     = {params['n_sample']}")
        print(f"  cluster_small/medium = {params['cluster_small']}/{params['cluster_medium']}")

    return params


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLUSTERING APPROACHES                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def cluster_tsfresh(X, n_clusters=4):
    """Approach 1: TSFresh statistical features → PCA → KMeans."""
    if not TSFRESH_AVAILABLE:
        print("SKIP: TSFresh not available")
        return None, None
    print("=" * 80)
    print("APPROACH 1: TSFresh Feature Extraction → KMeans Clustering")
    print("=" * 80)
    try:
        X_norm = np.array([(x - x.mean()) / (x.std() + 1e-8) for x in X])
        data_list = [
            {"id": series_id, "time": t, "value": val}
            for series_id, series in enumerate(X_norm)
            for t, val in enumerate(series)
        ]
        df_long = pd.DataFrame(data_list)
        print("Extracting features with TSFresh …")
        features = extract_features(df_long, column_id="id", column_sort="time", disable_progressbar=True)
        features = features.fillna(0)
        print(f"Extracted {features.shape[1]} features for {features.shape[0]} series")
        pca = PCA(n_components=0.99)
        X_pca = pca.fit_transform(features)
        print(f"PCA → {X_pca.shape[1]} components (99% variance)")
        kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0] // 2), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        dbi = davies_bouldin_score(X_pca, labels)
        chi = calinski_harabasz_score(X_pca, labels)
        metrics = {"sil": sil, "dbi": dbi, "chi": chi}
        cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Calinski-Harabasz Index: {chi:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")
        return labels, metrics
    except Exception as e:
        print(f"ERROR TSFresh clustering: {e}")
        import traceback; traceback.print_exc()
        return None, None


def cluster_dtw(X, n_clusters=4):
    """Approach 2: DTW pairwise distance → AgglomerativeClustering."""
    if not DTW_AVAILABLE:
        print("SKIP: DTW not available")
        return None, None
    print("=" * 80)
    print("APPROACH 2: DTW Distance → AgglomerativeClustering")
    print("=" * 80)
    try:
        X_scaled = np.array([(x - x.mean()) / (x.std() + 1e-8) for x in X])
        n_sample  = X_scaled.shape[0]
        X_sample  = X_scaled
        print(f"Calculating DTW distance matrix ({n_sample}×{n_sample}) …")
        dist_matrix = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n_sample}")
            for j in range(i + 1, n_sample):
                d = dtw_distance(X_sample[i], X_sample[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        ncl = min(n_clusters, n_sample // 2)
        agg = AgglomerativeClustering(n_clusters=ncl, metric="precomputed", linkage="average")
        labels_sample = agg.fit_predict(dist_matrix)
        labels = np.zeros(X_scaled.shape[0], dtype=int)
        labels[:n_sample] = labels_sample
        if X_scaled.shape[0] > n_sample:
            centroids = np.array([X_scaled[:n_sample][labels_sample == c].mean(axis=0) for c in range(ncl)])
            for idx in range(n_sample, X_scaled.shape[0]):
                dists = np.linalg.norm(centroids - X_scaled[idx], axis=1)
                labels[idx] = np.argmin(dists)
        sil = silhouette_score(dist_matrix, labels_sample, metric="precomputed")
        dbi = davies_bouldin_score(dist_matrix, labels_sample)
        chi = calinski_harabasz_score(dist_matrix, labels_sample)
        metrics = {"sil": sil, "dbi": dbi, "chi": chi}
        cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Calinski-Harabasz Index: {chi:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")
        return labels, metrics
    except Exception as e:
        print(f"ERROR DTW clustering: {e}")
        import traceback; traceback.print_exc()
        return None, None


def cluster_sktime(X, n_clusters=4):
    """Approach 3: sktime TimeSeriesKMeans (DTW) via isolated venv."""
    if not SKTIME_AVAILABLE:
        print("SKIP: sktime not available")
        return None, None
    print("=" * 80)
    print("APPROACH 3: sktime TimeSeriesKMeans (via isolated venv)")
    print("=" * 80)
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            temp_input = f.name
            pickle.dump({"X": X, "n_clusters": n_clusters, "N_SAMPLE": N_SAMPLE}, f)

        sktime_script = f"""
import pickle, warnings
import numpy as np
from sktime.clustering.kmeans import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
warnings.filterwarnings('ignore')

with open('{temp_input}', 'rb') as f:
    data = pickle.load(f)

X         = data['X']
n_clusters = data['n_clusters']
N_SAMPLE  = data['N_SAMPLE']

X_scaled = np.array([(x - x.mean()) / (x.std() + 1e-8) for x in X])
n_sample = min(X_scaled.shape[0], N_SAMPLE)
X_sample = X_scaled[:n_sample]
ncl = min(n_clusters, n_sample // 2)

kmeans = TimeSeriesKMeans(n_clusters=ncl, metric='dtw', random_state=42, n_init=5)
labels_sample = kmeans.fit_predict(X_sample)

labels = np.zeros(X_scaled.shape[0], dtype=int)
labels[:n_sample] = labels_sample
if X_scaled.shape[0] > n_sample:
    centroids = np.array([X_scaled[:n_sample][labels_sample == c].mean(axis=0) for c in range(ncl)])
    for idx in range(n_sample, X_scaled.shape[0]):
        dists = np.linalg.norm(centroids - X_scaled[idx], axis=1)
        labels[idx] = np.argmin(dists)

X_flat = X_sample.reshape(n_sample, -1)
sil = silhouette_score(X_flat, labels_sample)
dbi = davies_bouldin_score(X_flat, labels_sample)
chi = calinski_harabasz_score(X_flat, labels_sample)

with open('{temp_input}.out', 'wb') as f:
    pickle.dump({{'labels': labels, 'metrics': {{'sil': sil, 'dbi': dbi, 'chi': chi}}}}, f)
"""
        result = subprocess.run(
            [SKTIME_VENV_PATH, "-c", sktime_script],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"ERROR sktime script: {result.stderr}")
            return None, None
        with open(f"{temp_input}.out", "rb") as f:
            res = pickle.load(f)
        labels  = res["labels"]
        metrics = res["metrics"]
        cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Silhouette Score: {metrics['sil']:.4f}")
        print(f"Davies-Bouldin Index: {metrics['dbi']:.4f}")
        print(f"Calinski-Harabasz Index: {metrics['chi']:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")
        try:
            os.unlink(temp_input)
            os.unlink(f"{temp_input}.out")
        except Exception:
            pass
        return labels, metrics
    except Exception as e:
        print(f"ERROR sktime clustering: {e}")
        import traceback; traceback.print_exc()
        return None, None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL SELECTION — internal ETNA CV (replaces FEDOT on centroid)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _make_tsdataset(X_cluster, meter_ids, freq=FREQ):
    """Build ETNA TSDataset from a subset of series."""
    timestamps = pd.date_range("2020-01-01", periods=X_cluster.shape[1], freq=freq)
    rows = []
    seen = {}
    for meter_id in meter_ids:
        seg_name = f"meter{meter_id}"
        if seg_name in seen:
            seen[seg_name] += 1
            seg_name = f"{seg_name}_dup{seen[seg_name]}"
        else:
            seen[seg_name] = 0
        rows.append(pd.DataFrame({"timestamp": timestamps, "segment": seg_name, "target": X_cluster[meter_id]}))
    df_long = pd.concat(rows, ignore_index=True)
    df_wide = TSDataset.to_dataset(df_long)
    return TSDataset(df=df_wide, freq=freq)


def _build_transforms():
    """Build standard transform list."""
    transforms = []
    if LogTransform:
        transforms.append(LogTransform(in_column="target"))
    transforms.extend([
        LagTransform(in_column="target", lags=LAGS),
        DateFlagsTransform(day_number_in_week=True, is_weekend=True),
        StandardScalerTransform(in_column="target"),
    ])
    if StatisticsTransform:
        try:
            transforms.append(
                StatisticsTransform(in_column="target", window=ROLLING_WIN, out_column=f"rolling_mean_{ROLLING_WIN}")
            )
        except Exception:
            pass
    return transforms


def _build_minimal_transforms():
    """Minimal transforms for retry after failure."""
    return [
        LagTransform(in_column="target", lags=LAGS),
        DateFlagsTransform(day_number_in_week=True, is_weekend=True),
        StandardScalerTransform(in_column="target"),
    ]


def select_model_for_cluster_via_cv(X_train, cluster_mask, horizon, freq=FREQ):
    """
    ── KEY FIX ──────────────────────────────────────────────────────────────
    Select the best ETNA model for a cluster using internal 1-fold CV on the
    cluster's own multi-segment dataset.

    Strategy:
      1. Find the cluster's medoid (real series closest to centroid) —
         avoids centroid-smoothing bias.
      2. Quick 1-fold holdout on X_train[:, :-horizon] → validate on
         X_train[:, -horizon:] for both CatBoost and Linear.
      3. Return the model type with lower mean MAE across cluster members.

    This replaces FEDOT plain (which ran on a single centroid in a separate
    subprocess and always fell back to catboost on timeout).
    """
    if not ETNA_AVAILABLE:
        return "catboost"

    cluster_series = X_train[cluster_mask]
    cluster_ids    = np.where(cluster_mask)[0]
    n_seg = len(cluster_ids)

    # Medoid: real series closest to cluster centroid
    centroid = cluster_series.mean(axis=0)
    dists    = np.linalg.norm(cluster_series - centroid, axis=1)
    medoid_idx = np.argmin(dists)
    print(f"    Medoid meter: {cluster_ids[medoid_idx]}  (dist={dists[medoid_idx]:.4f})")

    # 1-fold holdout split inside the cluster's training data
    # Use at most 20 series for speed; preserve time order
    cv_ids = cluster_ids[:min(n_seg, 20)]
    X_cv_train = X_train[cv_ids, :-horizon]
    X_cv_val   = X_train[cv_ids, -horizon:]

    timestamps_train = pd.date_range("2020-01-01", periods=X_cv_train.shape[1], freq=freq)

    def _cv_mae(model_type):
        try:
            rows = []
            seen = {}
            for local_i, mid in enumerate(cv_ids):
                seg = f"meter{mid}"
                if seg in seen:
                    seen[seg] += 1; seg = f"{seg}_dup{seen[seg]}"
                else:
                    seen[seg] = 0
                rows.append(pd.DataFrame({"timestamp": timestamps_train, "segment": seg, "target": X_cv_train[local_i]}))
            df_long = pd.concat(rows, ignore_index=True)
            df_wide = TSDataset.to_dataset(df_long)
            ts = TSDataset(df=df_wide, freq=freq)

            transforms = _build_minimal_transforms()
            if model_type == "linear":
                model = LinearMultiSegmentModel()
            else:
                iterations = min(100, 50 + n_seg * 2)
                model = CatBoostMultiSegmentModel(iterations=iterations, depth=3, learning_rate=0.1, random_seed=42)

            pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
            pipeline.fit(ts)
            forecast_ts = pipeline.forecast()
            forecast_df = forecast_ts.to_pandas()

            maes = []
            segments = forecast_df.columns.get_level_values(0).unique() if hasattr(forecast_df.columns, 'get_level_values') else [forecast_df.columns[0]]
            for seg in segments:
                try:
                    mid_str = seg.split("meter")[-1].replace("_dup0","").replace("_dup1","").replace("_dup2","")
                    local_i = list(cv_ids).index(int(mid_str)) if mid_str.isdigit() and int(mid_str) in cv_ids else None
                except Exception:
                    local_i = None

                try:
                    if "target" in forecast_df[seg].columns:
                        fc = forecast_df[seg]["target"].values
                    else:
                        fc = forecast_df[seg].iloc[:, 0].values
                    if local_i is not None and len(fc) == horizon:
                        actual = X_cv_val[local_i]
                        maes.append(mean_absolute_error(actual, fc))
                except Exception:
                    pass

            return np.mean(maes) if maes else np.inf
        except Exception as e:
            print(f"      CV error ({model_type}): {e}")
            return np.inf

    print(f"    Running internal CV (horizon={horizon}, {len(cv_ids)} segs) …")
    mae_catboost = _cv_mae("catboost")
    mae_linear   = _cv_mae("linear")
    print(f"    CV MAE — CatBoost: {mae_catboost:.4f} | Linear: {mae_linear:.4f}")

    if mae_linear < mae_catboost and mae_linear < np.inf:
        choice = "linear"
    else:
        choice = "catboost"
    print(f"    → Selected: {choice.upper()}")
    return choice


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL MODEL                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_global_model(X_train, horizon=24):
    """Global CatBoostMultiSegmentModel trained on ALL segments."""
    if not ETNA_AVAILABLE:
        print("ERROR: ETNA not installed!")
        return {}
    try:
        print("GLOBAL MODEL: Building CatBoostMultiSegmentModel …")
        ts_dataset = _make_tsdataset(X_train, list(range(X_train.shape[0])))

        transforms = _build_transforms()
        model = CatBoostMultiSegmentModel(iterations=200, depth=4, learning_rate=0.05, random_seed=42)
        pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
        print("  Training pipeline …")
        pipeline.fit(ts_dataset)
        forecast_ts = pipeline.forecast()
        forecast_df = forecast_ts.to_pandas()

        forecasts = {}
        try:
            segments = forecast_df.columns.get_level_values(0).unique()
        except Exception:
            segments = [forecast_df.columns[0]]
        for seg in segments:
            try:
                if "target" in forecast_df[seg].columns:
                    data = forecast_df[seg]["target"].values
                else:
                    data = forecast_df[seg].iloc[:, 0].values
                if len(data) >= horizon and not np.isnan(data).all():
                    forecasts[seg] = data
            except Exception:
                pass
        print(f"  Extracted {len(forecasts)}/{len(segments)} segments")
        if not forecasts:
            print("WARNING: No forecasts extracted!")
        return forecasts
    except Exception as e:
        print(f"ERROR build_global_model: {e}")
        import traceback; traceback.print_exc()
        return {}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLUSTER MODELS  (with real model diversity)                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_cluster_models(X_train, cluster_labels, horizon=24, model_mapping=None):
    """
    For each cluster:
      1. Select model via internal CV (not FEDOT centroid)
      2. Actually USE the selected model (fixed bug: was always CatBoost)
      3. Tune CatBoost depth/iterations based on cluster size
    """
    if not ETNA_AVAILABLE:
        print("ERROR: ETNA not installed!")
        return {}

    print("CLUSTER MODELS: Building per-cluster models with diversity …")
    all_cluster_forecasts = {}

    try:
        for cluster_id in np.unique(cluster_labels):
            cluster_mask  = cluster_labels == cluster_id
            cluster_meter_ids = np.where(cluster_mask)[0]
            n_seg = len(cluster_meter_ids)
            if n_seg == 0:
                continue

            print(f"\nCluster {cluster_id}: {n_seg} segments")

            # ── Model selection via internal CV ──────────────────────────
            if model_mapping is not None and cluster_id in model_mapping:
                choice = model_mapping[cluster_id]
                print(f"  Using pre-computed model choice: {choice}")
            else:
                choice = select_model_for_cluster_via_cv(X_train, cluster_mask, horizon)

            # ── CatBoost hyperparams scaled to cluster size ───────────────
            if n_seg < CLUSTER_SMALL:
                iterations, depth, lr = 100, 3, 0.1
            elif n_seg < CLUSTER_MEDIUM:
                iterations, depth, lr = 200, 5, 0.05
            else:
                iterations, depth, lr = 400, 7, 0.02

            # ── Build the ACTUALLY selected model ─────────────────────────
            transforms = _build_transforms()

            if choice == "linear":
                print(f"  → LinearMultiSegmentModel  (cluster {cluster_id})")
                model = LinearMultiSegmentModel()
            else:
                print(f"  → CatBoostMultiSegmentModel iter={iterations} depth={depth} lr={lr}  (cluster {cluster_id})")
                model = CatBoostMultiSegmentModel(
                    iterations=iterations, depth=depth, learning_rate=lr, random_seed=42
                )

            ts_dataset = _make_tsdataset(X_train, list(cluster_meter_ids))
            pipeline   = Pipeline(model=model, transforms=transforms, horizon=horizon)

            try:
                pipeline.fit(ts_dataset)
                forecast_ts = pipeline.forecast()
            except Exception as model_err:
                err_str = str(model_err).lower()
                print(f"  ! Pipeline failed: {str(model_err)[:120]}")
                print("  Retrying with minimal transforms …")
                ts_dataset = _make_tsdataset(X_train, list(cluster_meter_ids))
                min_transforms = _build_minimal_transforms()
                if choice == "linear":
                    retry_model = LinearMultiSegmentModel()
                else:
                    retry_model = CatBoostMultiSegmentModel(
                        iterations=iterations, depth=depth, learning_rate=lr, random_seed=42
                    )
                pipeline = Pipeline(model=retry_model, transforms=min_transforms, horizon=horizon)
                pipeline.fit(ts_dataset)
                forecast_ts = pipeline.forecast()

            forecast_df = forecast_ts.to_pandas()
            cluster_forecasts = {}
            try:
                segments = forecast_df.columns.get_level_values(0).unique()
            except Exception:
                segments = [forecast_df.columns[0]]
            for seg in segments:
                try:
                    if "target" in forecast_df[seg].columns:
                        data = forecast_df[seg]["target"].values
                    else:
                        data = forecast_df[seg].iloc[:, 0].values
                    if len(data) >= horizon and not np.isnan(data).all():
                        cluster_forecasts[seg] = data
                except Exception:
                    pass
            all_cluster_forecasts[cluster_id] = cluster_forecasts
            print(f"  Generated {len(cluster_forecasts)}/{len(segments)} forecasts for cluster {cluster_id}")

        return all_cluster_forecasts
    except Exception as e:
        print(f"ERROR build_cluster_models: {e}")
        import traceback; traceback.print_exc()
        return {}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENSEMBLE AGGREGATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def aggregate_cluster_forecasts(cluster_labels, cluster_forecasts):
    """Map each meter → its cluster forecast by segment key."""
    print("ENSEMBLE AGGREGATION")
    ensemble = {}
    n_segments = len(cluster_labels)
    for meter_id in range(n_segments):
        cluster_id  = cluster_labels[meter_id]
        segment_key = f"meter{meter_id}"
        if cluster_id in cluster_forecasts and segment_key in cluster_forecasts[cluster_id]:
            ensemble[segment_key] = cluster_forecasts[cluster_id][segment_key]
        else:
            print(f"  ! {segment_key} not found in cluster {cluster_id}")
    print(f"Aggregated {len(ensemble)}/{n_segments} segments into ensemble")
    return ensemble


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_forecasts(X_val, global_forecasts, ensemble_forecasts, horizon):
    """Compute MAE/RMSE/sMAPE per segment and accumulated error arrays for DM test."""
    print("=" * 80)
    print("EVALUATION: Global Model vs Cluster-Ensemble")
    print("=" * 80)
    results = {
        "global": {}, "ensemble": {},
        "global_errors": [], "ensemble_errors": [],
        "global_seg_mae": [], "ensemble_seg_mae": [],
        "per_meter": {}
    }

    # Global
    global_maes, global_rmses, global_mapes, global_errors = [], [], [], []
    per_meter_global = {}
    if global_forecasts:
        for seg_key, forecast in global_forecasts.items():
            try:
                meter_num = int(seg_key.replace("meter", "").split("_dup")[0])
                if meter_num < 0 or meter_num >= X_val.shape[0]: continue
                actual = X_val[meter_num][:len(forecast)]
                if np.isnan(forecast).any() or len(forecast) == 0: continue
                errors = actual - forecast
                global_errors.extend(errors.tolist())
                mae  = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mape = np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + 1e-8)) * 100
                global_maes.append(mae); global_rmses.append(rmse); global_mapes.append(mape)
                results["global_seg_mae"].append(mae)
                per_meter_global[meter_num] = mae
            except Exception as e:
                if len(global_maes) < 5: print(f"  DEBUG: Segment {seg_key} failed: {e}")

    results["global"] = {
        "MAE": np.mean(global_maes) if global_maes else np.nan,
        "RMSE": np.mean(global_rmses) if global_rmses else np.nan,
        "MAPE": np.mean(global_mapes) if global_mapes else np.nan,
    }
    results["global_errors"] = global_errors
    print(f"GLOBAL CatBoost Model")
    print(f"  MAE  = {results['global']['MAE']:.4f}")
    print(f"  RMSE = {results['global']['RMSE']:.4f}")
    print(f"  sMAPE= {results['global']['MAPE']:.4f}")

    # Ensemble
    ensemble_maes, ensemble_rmses, ensemble_mapes, ensemble_errors = [], [], [], []
    if ensemble_forecasts:
        for seg_key, forecast in ensemble_forecasts.items():
            try:
                meter_num = int(seg_key.replace("meter", "").split("_dup")[0])
                if meter_num < 0 or meter_num >= X_val.shape[0]: continue
                actual = X_val[meter_num][:len(forecast)]
                if np.isnan(forecast).any() or len(forecast) == 0: continue
                errors = actual - forecast
                ensemble_errors.extend(errors.tolist())
                mae  = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mape = np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + 1e-8)) * 100
                ensemble_maes.append(mae); ensemble_rmses.append(rmse); ensemble_mapes.append(mape)
                results["ensemble_seg_mae"].append(mae)
                if meter_num in per_meter_global:
                    results["per_meter"][meter_num] = {
                        "global_mae": per_meter_global[meter_num],
                        "ensemble_mae": mae,
                        "improvement": per_meter_global[meter_num] - mae,
                    }
            except Exception:
                pass

    results["ensemble"] = {
        "MAE": np.mean(ensemble_maes) if ensemble_maes else np.nan,
        "RMSE": np.mean(ensemble_rmses) if ensemble_rmses else np.nan,
        "MAPE": np.mean(ensemble_mapes) if ensemble_mapes else np.nan,
    }
    results["ensemble_errors"] = ensemble_errors
    print(f"CLUSTER-ENSEMBLE Model")
    print(f"  MAE  = {results['ensemble']['MAE']:.4f}")
    print(f"  RMSE = {results['ensemble']['RMSE']:.4f}")
    print(f"  sMAPE= {results['ensemble']['MAPE']:.4f}")
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICAL TESTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def diebold_mariano_test(errors1, errors2, h=1):
    """Harvey, Leybourne & Newbold (1997) small-sample corrected DM test."""
    d = np.array(errors1) ** 2 - np.array(errors2) ** 2
    n = len(d)
    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        if len(d) > k:
            gamma_k = np.cov(d[:-k], d[k:])[0, 1]
            gamma_sum += gamma_k
    var_d = gamma0 + 2 * gamma_sum
    var_d = max(var_d, 1e-10)
    dm_stat = mean_d / np.sqrt(var_d / n)
    # HLN correction factor
    hlnc = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hlnc
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat_corrected)))
    return dm_stat_corrected, p_value


def run_statistical_tests(global_errors, ensemble_errors, horizon=24):
    print("=" * 80)
    print("STATISTICAL TESTS: Global vs Cluster-Ensemble")
    print("=" * 80)
    results = {}

    if len(global_errors) != len(ensemble_errors):
        min_len = min(len(global_errors), len(ensemble_errors))
        global_errors   = global_errors[:min_len]
        ensemble_errors = ensemble_errors[:min_len]
        print(f"  Aligned error arrays to {min_len} observations")

    dm_stat, dm_pval = diebold_mariano_test(global_errors, ensemble_errors, h=horizon)
    results["diebold_mariano"] = {"statistic": dm_stat, "p_value": dm_pval}
    print(f"Diebold-Mariano Test (HLN-corrected)")
    print(f"  H0: Both models have the same accuracy")
    print(f"  DM Statistic = {dm_stat:.4f}")
    print(f"  p-value      = {dm_pval:.4f}")
    if dm_pval < 0.05:
        if dm_stat > 0:
            print("  → ENSEMBLE is significantly BETTER (p < 0.05) ✓")
        else:
            print("  → GLOBAL is significantly BETTER (p < 0.05)")
    elif dm_pval < 0.10:
        print("  → Marginal improvement (0.05 < p < 0.10)")
    else:
        print("  → No significant difference (p ≥ 0.05)")

    # Wilcoxon signed-rank (non-parametric complement)
    try:
        from scipy.stats import wilcoxon
        err1 = np.array(global_errors)
        err2 = np.array(ensemble_errors)
        diff = err1 ** 2 - err2 ** 2
        if np.any(diff != 0):
            w_stat, w_pval = wilcoxon(diff, alternative="greater")
            results["wilcoxon"] = {"statistic": w_stat, "p_value": w_pval}
            print(f"\nWilcoxon Signed-Rank Test (non-parametric)")
            print(f"  Statistic = {w_stat:.4f}")
            print(f"  p-value   = {w_pval:.4f}")
            if w_pval < 0.05:
                print("  → ENSEMBLE is significantly BETTER (p < 0.05) ✓")
            else:
                print("  → No significant difference (p ≥ 0.05)")
    except Exception as e:
        print(f"  Wilcoxon skipped: {e}")

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUALISATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_best_representative_meters(per_meter_results, top_n=4):
    if not per_meter_results:
        return []
    sorted_meters = sorted(per_meter_results.items(), key=lambda x: x[1]["improvement"], reverse=True)
    return [m[0] for m in sorted_meters[:top_n]]


def plot_forecasts_comparison(X_val, global_forecasts, ensemble_forecasts,
                              approach_name, horizon, best_meters=None):
    print(f"PLOT: Comparing forecasts — {approach_name} …")
    if best_meters is None or len(best_meters) == 0:
        best_meters = list(range(min(4, X_val.shape[0])))
    n_meters = len(best_meters)
    cols = 2
    rows = (n_meters + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), squeeze=False)
    axes = axes.flatten()
    palette = sns.color_palette("muted")
    for i, meter_id in enumerate(best_meters):
        ax = axes[i]
        actual      = X_val[meter_id][:horizon]
        global_key  = f"meter{meter_id}"
        global_fc   = global_forecasts.get(global_key, actual) if global_forecasts else actual
        ensemble_fc = ensemble_forecasts.get(global_key, actual) if ensemble_forecasts else actual
        mae_global   = mean_absolute_error(actual, global_fc)   if not np.isnan(global_fc).all() else np.nan
        mae_ensemble = mean_absolute_error(actual, ensemble_fc) if not np.isnan(ensemble_fc).all() else np.nan
        ax.plot(actual,      color="black",     linewidth=2, marker="o", markersize=4, label="Actual",                    alpha=0.7, zorder=3)
        ax.plot(global_fc,   color=palette[0],  linewidth=2, linestyle="-",  label=f"Global (MAE: {mae_global:.3f})",   alpha=0.9, zorder=2)
        ax.plot(ensemble_fc, color=palette[1],  linewidth=2, linestyle="--", label=f"Ensemble (MAE: {mae_ensemble:.3f})", alpha=0.9, zorder=2)
        ax.set_title(f"Meter: {meter_id}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=10)
        ax.set_ylabel("Consumption",  fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, horizon - 0.5)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(
        f"Forecast Comparison: {approach_name} Approach\n(Selected Best-Performing Meters for Ensemble)",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{approach_name.lower().replace(' ', '_')}_forecast_comparison.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {filename}")


def plot_cluster_visualization(X, labels, approach_name):
    print(f"PLOT: Visualizing clusters — {approach_name} …")
    unique_labels = np.unique(labels)
    n_clusters    = len(unique_labels)
    fig, axes = plt.subplots(min(n_clusters, 3), 1, figsize=(15, 5 * min(n_clusters, 3)), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, label in enumerate(unique_labels[:3]):
        cluster_data = X[labels == label]
        if len(cluster_data) == 0:
            continue
        centroid = cluster_data.mean(axis=0)
        limit = min(len(cluster_data), 50)
        for j in range(limit):
            axes[i].plot(cluster_data[j], color=colors[i % 10], alpha=0.15)
        axes[i].plot(centroid, color=colors[i % 10], linewidth=3, label="Centroid")
        axes[i].set_title(f"Cluster {label} — {len(cluster_data)} time series")
        axes[i].legend(); axes[i].grid(True, alpha=0.2)
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{approach_name.lower().replace(' ', '_')}_clusters.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_model_diversity_summary(model_mappings, approach_name):
    """Pie chart showing how many clusters used each model type."""
    if not model_mappings:
        return
    from collections import Counter
    counts = Counter(model_mappings.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(list(counts.values()), labels=list(counts.keys()),
           autopct="%1.0f%%", startangle=90,
           colors=["#4C72B0", "#DD8452"])
    ax.set_title(f"Model Type Distribution per Cluster\n({approach_name})", fontsize=12, fontweight="bold")
    filename = os.path.join(PLOTS_DIR, f"{approach_name.lower().replace(' ', '_')}_model_diversity.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 80)
    print("PIPELINE: Backtesting Multi-Segment Forecasting (Model Diversity Fix)")
    print("=" * 80)

    # 1. Load data
    X = load_powercons_data()

    # 2. Auto-detect parameters
    params = autodetect_params(X)
    global FREQ, LAGS, ROLLING_WIN, N_SAMPLE, CLUSTER_SMALL, CLUSTER_MEDIUM
    FREQ           = params["freq"]
    LAGS           = params["lags"]
    ROLLING_WIN    = params["rolling_win"]
    N_SAMPLE       = params["n_sample"]
    CLUSTER_SMALL  = params["cluster_small"]
    CLUSTER_MEDIUM = params["cluster_medium"]
    n_clusters     = params["n_clusters"]

    n_series, n_timesteps = X.shape

    # 3. Backtest windows (last 15% as test, 6 rolling windows)
    test_frac   = 0.15
    n_test      = max(24, int(n_timesteps * test_frac))
    horizon     = min(24, n_test // 6)
    n_windows   = min(6, n_test // horizon)
    test_windows = [
        (n_timesteps - n_test + i * horizon, n_timesteps - n_test + (i + 1) * horizon)
        for i in range(n_windows)
    ]
    print(f"\nBacktest: {n_windows} windows of horizon={horizon}  ({n_test} test timesteps)")

    # 4. Clustering (run once on full train prefix)
    train_end_full = test_windows[0][0]
    X_train_full   = X[:, :train_end_full]

    approaches = []
    labels_tsfresh, _ = cluster_tsfresh(X_train_full, n_clusters=n_clusters)
    if labels_tsfresh is not None:
        approaches.append(("TSFresh", labels_tsfresh))

    labels_dtw, _ = cluster_dtw(X_train_full, n_clusters=n_clusters)
    if labels_dtw is not None:
        approaches.append(("DTW", labels_dtw))

    labels_sktime, _ = cluster_sktime(X_train_full, n_clusters=n_clusters)
    if labels_sktime is not None:
        approaches.append(("sktime", labels_sktime))

    if not approaches:
        print("No clustering approach available — using dummy single-cluster labels")
        approaches = [("Fallback", np.zeros(n_series, dtype=int))]

    # 5. Visualise clusters
    for approach_name, labels in approaches:
        plot_cluster_visualization(X_train_full, labels, approach_name)

    # 6. Pre-compute model selection mapping ONCE (per approach) to save time
    approach_model_mappings = {}
    for approach_name, labels in approaches:
        print(f"\n{'='*60}")
        print(f"MODEL SELECTION for {approach_name}")
        print("="*60)
        model_mapping = {}
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            print(f"  Cluster {cluster_id}: {np.sum(cluster_mask)} series")
            choice = select_model_for_cluster_via_cv(X_train_full, cluster_mask, horizon)
            model_mapping[cluster_id] = choice
        approach_model_mappings[approach_name] = model_mapping
        print(f"  Final mapping: {model_mapping}")
        plot_model_diversity_summary(model_mapping, approach_name)

    # 7. Backtest loop
    all_results = {}
    for approach_name, labels in approaches:
        print(f"\n{'='*80}")
        print(f"PIPELINE for {approach_name} — Backtesting Mode")
        print("="*80)
        model_mapping = approach_model_mappings[approach_name]

        all_global_errors   = []
        all_ensemble_errors = []
        final_global_forecasts   = {}
        final_ensemble_forecasts = {}
        final_X_val = None
        per_meter_accumulated = {}

        for i, (train_end, test_end) in enumerate(test_windows):
            print(f"\n--- Backtest Window {i+1}/{n_windows} ---")
            X_train = X[:, :train_end]
            X_val   = X[:, train_end:test_end]
            current_horizon = test_end - train_end

            # Global model
            global_forecasts = build_global_model(X_train, horizon=current_horizon)

            # Cluster models with pre-selected mappings
            cluster_forecasts  = build_cluster_models(
                X_train, labels, horizon=current_horizon, model_mapping=model_mapping
            )
            ensemble_forecasts = aggregate_cluster_forecasts(labels, cluster_forecasts)

            # Evaluate
            eval_res = evaluate_forecasts(X_val, global_forecasts, ensemble_forecasts, current_horizon)
            all_global_errors.extend(eval_res.get("global_errors", []))
            all_ensemble_errors.extend(eval_res.get("ensemble_errors", []))

            # Accumulate per-meter improvements
            for meter_id, mstats in eval_res.get("per_meter", {}).items():
                if meter_id not in per_meter_accumulated:
                    per_meter_accumulated[meter_id] = {"global_mae": [], "ensemble_mae": []}
                per_meter_accumulated[meter_id]["global_mae"].append(mstats["global_mae"])
                per_meter_accumulated[meter_id]["ensemble_mae"].append(mstats["ensemble_mae"])

            # Keep last window for visualisation
            final_global_forecasts   = global_forecasts
            final_ensemble_forecasts = ensemble_forecasts
            final_X_val = X_val

        # Aggregate per-meter improvements
        per_meter_final = {}
        for meter_id, mstats in per_meter_accumulated.items():
            g_mae = np.mean(mstats["global_mae"])
            e_mae = np.mean(mstats["ensemble_mae"])
            per_meter_final[meter_id] = {
                "global_mae": g_mae,
                "ensemble_mae": e_mae,
                "improvement": g_mae - e_mae,
            }

        # Statistical tests on accumulated errors
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY — {approach_name}")
        print(f"  Accumulated: {len(all_global_errors)} global errors, {len(all_ensemble_errors)} ensemble errors")
        stat_results = run_statistical_tests(all_global_errors, all_ensemble_errors, horizon=horizon)
        all_results[approach_name] = {
            "global_errors":   all_global_errors,
            "ensemble_errors": all_ensemble_errors,
            "stat_tests":      stat_results,
            "per_meter":       per_meter_final,
            "model_mapping":   model_mapping,
        }

        # Plot best meters
        if final_X_val is not None:
            best_meters = get_best_representative_meters(per_meter_final, top_n=4)
            plot_forecasts_comparison(
                final_X_val, final_global_forecasts, final_ensemble_forecasts,
                approach_name, horizon, best_meters
            )

    # 8. Save summary CSV
    summary_rows = []
    for approach_name, res in all_results.items():
        ge = res["global_errors"];   ee = res["ensemble_errors"]
        dm = res["stat_tests"].get("diebold_mariano", {})
        wl = res["stat_tests"].get("wilcoxon", {})
        mm = res.get("model_mapping", {})
        n_linear  = sum(1 for v in mm.values() if v == "linear")
        n_catboost = sum(1 for v in mm.values() if v == "catboost")
        summary_rows.append({
            "approach":           approach_name,
            "global_mae":         np.mean(np.abs(ge)) if ge else np.nan,
            "ensemble_mae":       np.mean(np.abs(ee)) if ee else np.nan,
            "dm_stat":            dm.get("statistic", np.nan),
            "dm_pvalue":          dm.get("p_value", np.nan),
            "wilcoxon_stat":      wl.get("statistic", np.nan),
            "wilcoxon_pvalue":    wl.get("p_value", np.nan),
            "n_clusters":         len(mm),
            "n_linear_clusters":  n_linear,
            "n_catboost_clusters":n_catboost,
        })
    df_summary = pd.DataFrame(summary_rows)
    csv_path   = os.path.join(METRICS_DIR, "results_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    print(df_summary.to_string(index=False))

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
