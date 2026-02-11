"""
The Project: Energy Consumption Clustering and Forecasting
- Using TSFresh for statistical features and clustering
- Trying out DTW for time-series distance matrices
- Using sktime and ETNA's CatBoost model for forecasting
- Building an ensemble of cluster-specific models to see if it beats the global baseline
"""

import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import zipfile
import subprocess
import pickle
import tempfile
from urllib.request import urlretrieve
from urllib.error import URLError

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform

print("\n" + "="*80)
print("CHECKING IMPORTS")
print("="*80)

# DTW Distance
try:
    from dtaidistance.dtw import distance as dtw_distance
    DTW_AVAILABLE = True
    print("[+] dtaidistance (dtw.distance)")
except ImportError as e:
    print(f"[-] dtaidistance: {e}")
    DTW_AVAILABLE = False

# TSFresh
try:
    from tsfresh import extract_features
    TSFRESH_AVAILABLE = True
    print("[+] TSFresh")
except ImportError as e:
    print(f"[-] TSFresh: {e}")
    TSFRESH_AVAILABLE = False

# sktime via separate venv
SKTIME_VENV_PATH = os.environ.get("SKTIME_VENV_PATH", "/opt/sktime_env/bin/python")

# fallback for docker
if not os.path.exists(SKTIME_VENV_PATH):
    import sys
    SKTIME_VENV_PATH = sys.executable

try:
    result = subprocess.run(
        [SKTIME_VENV_PATH, '-c', 'import sktime; print("OK")'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode == 0 and 'OK' in result.stdout:
        SKTIME_AVAILABLE = True
        print(f"[+] sktime (via {SKTIME_VENV_PATH})")
    else:
        SKTIME_AVAILABLE = False
        print(f"[-] sktime check failed: {result.stderr}")
        
except Exception as e:
    print(f"[-] sktime: {e}")
    SKTIME_AVAILABLE = False

# ETNA + Models + Transforms
try:
    from etna.models import CatBoostMultiSegmentModel
    from etna.pipeline import Pipeline
    from etna.datasets import TSDataset
    from etna.transforms import (LagTransform, DateFlagsTransform, 
                                 StandardScalerTransform)
    
    # Try importing advanced transforms individually for better error reporting
    try:
        from etna.transforms import LogTransform
    except ImportError:
        LogTransform = None
        print("[-] Notice: LogTransform not found in ETNA")
        
    try:
        from etna.transforms import MeanTransform as StatisticsTransform
    except ImportError:
        try:
            from etna.transforms import StatisticsTransform
        except ImportError:
            StatisticsTransform = None
            print("[-] Notice: StatisticsTransform/MeanTransform not found in ETNA")

    ETNA_AVAILABLE = True
    print("[+] ETNA + CatBoostMultiSegmentModel")
except Exception as e:
    print(f"[-] ETNA Import Error: {e}")
    import traceback
    traceback.print_exc()
    ETNA_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# ============================================================================
# DIRECTORIES
# ============================================================================

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
DATA_DIR = "data"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def download_powercons_dataset(data_dir=DATA_DIR):
    """Download UCR PowerCons dataset"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    archive_url = (
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        "UCRArchive_2018.zip"
    )
    archive_zip_path = os.path.join(data_dir, "UCRArchive_2018.zip")
    dataset_dir = os.path.join(data_dir, "PowerCons")

    if os.path.exists(dataset_dir):
        return True

    try:
        print("Downloading UCR Archive...")
        urlretrieve(archive_url, archive_zip_path)
    except URLError as e:
        print(f"Download error: {e}")
        if os.path.exists(archive_zip_path):
            os.remove(archive_zip_path)
        return False

    try:
        print("Extracting PowerCons files...")
        with zipfile.ZipFile(archive_zip_path, "r") as zf:
            for f in zf.namelist():
                if f.startswith("UCRArchive_2018/PowerCons/"):
                    zf.extract(f, data_dir)
    except Exception as e:
        print(f"Extraction error: {e}")
        if os.path.exists(archive_zip_path):
            os.remove(archive_zip_path)
        return False

    try:
        nested_dir = os.path.join(data_dir, "UCRArchive_2018", "PowerCons")
        if os.path.exists(nested_dir):
            shutil.move(nested_dir, data_dir)
        root_ucr = os.path.join(data_dir, "UCRArchive_2018")
        if os.path.exists(root_ucr):
            shutil.rmtree(root_ucr)
    except Exception as e:
        print(f"File organization error: {e}")
        return False

    try:
        os.remove(archive_zip_path)
    except Exception:
        pass

    print("Dataset ready!")
    return True

def load_powercons_data(data_dir=DATA_DIR):
    """Load PowerCons dataset"""
    print("Loading PowerCons dataset...")
    if not download_powercons_dataset(data_dir):
        print("[FALLBACK] Using synthetic data...")
        return load_synthetic_data()

    dataset_dir = os.path.join(data_dir, "PowerCons")
    train_path = os.path.join(dataset_dir, "PowerCons_TRAIN.tsv")
    test_path = os.path.join(dataset_dir, "PowerCons_TEST.tsv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: dataset files not found")
        return load_synthetic_data()

    train = pd.read_csv(train_path, header=None, sep="\t")
    test = pd.read_csv(test_path, header=None, sep="\t")

    X_train = train.iloc[:, 1:].values
    X_test = test.iloc[:, 1:].values
    X = np.vstack([X_train, X_test]).astype(float)

    print(f"PowerCons loaded: {X.shape[0]} series, {X.shape[1]} timesteps")
    return X

def load_synthetic_data():
    """Generate synthetic data"""
    print("Generating synthetic PowerCons data...")
    np.random.seed(42)
    n_series = 30  # Small count for clustering parameters
    n_timesteps = 144

    X = np.zeros((n_series, n_timesteps))
    for i in range(n_series):
        t = np.arange(n_timesteps)
        X[i] = (
            2 + 1.5 * np.sin(2 * np.pi * t / 24) +
            0.3 * np.sin(2 * np.pi * t / 168) +
            0.05 * t +
            np.random.normal(0, 0.2, n_timesteps)
        )

    print(f"Synthetic data: {X.shape[0]} series x {X.shape[1]} timesteps")
    return X

# ============================================================================
# APPROACH 1: TSFresh Feature Extraction + KMeans Clustering
# ============================================================================

def cluster_tsfresh(X, n_clusters=4):
    """
    Approach 1: Clustering based on TSFresh statistical features.
    
    1. Turn series into long format (id, time, value)
    2. Run extract_features to get a bunch of numeric stats
    3. Run KMeans on the features to group similar meters
    """
    if not TSFRESH_AVAILABLE:
        print("[SKIP] TSFresh not available")
        return None, None

    print("\n" + "="*80)
    print("APPROACH 1: TSFresh Feature Extraction + KMeans Clustering")
    print("="*80)

    try:
        print("Transforming series to long format...")
        data_list = []
        for series_id, series in enumerate(X):
            for t, val in enumerate(series):
                data_list.append({'id': series_id, 'time': t, 'value': val})
        df_long = pd.DataFrame(data_list)

        print("Extracting features with TSFresh...")
        features = extract_features(
            df_long,
            column_id='id',
            column_sort='time',
            disable_progressbar=True
        )
        features = features.fillna(0)

        print(f"Extracted {features.shape[1]} features for {features.shape[0]} series")

        print("PCA for dimensionality reduction...")
        pca = PCA(n_components=0.99)
        X_pca = pca.fit_transform(features)
        print(f"PCA: {X_pca.shape[1]} components explain 99% of variance")

        print("KMeans feature clustering...")
        kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]//2), 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        dbi = davies_bouldin_score(X_pca, labels)
        chi = calinski_harabasz_score(X_pca, labels)

        metrics = {'sil': sil, 'dbi': dbi, 'chi': chi}

        cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Calinski-Harabasz Index: {chi:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")

        return labels, metrics
    except Exception as e:
        print(f"[ERROR] TSFresh clustering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# APPROACH 2: DTW Distance + KMeans Clustering
# ============================================================================

def cluster_dtw(X, n_clusters=4):
    """
    Approach 2: Clustering with Dynamic Time Warping (DTW) distances.
    
    1. Normalize each series so they are comparable (z-score for each series)
    2. Build a distance matrix using DTW (takes a while, so subsampling)
    3. Cluster the meters based on their DTW shapes
    """
    if not DTW_AVAILABLE:
        print("[SKIP] DTW not available")
        return None, None
        
    print("\n" + "="*80)
    print("APPROACH 2: DTW Distance + KMeans Clustering")
    print("="*80)

    try:
        print("Standardizing series (z-score for each series)...")
        X_scaled = np.array([((x - x.mean()) / (x.std() + 1e-8)) for x in X])

        # Subsample for speed
        n_sample = min(X_scaled.shape[0], 50)
        X_sample = X_scaled[:n_sample]
        
        print(f"Calculating DTW distance matrix ({n_sample}x{n_sample})...")
        dist_matrix = np.zeros((n_sample, n_sample))

        for i in range(n_sample):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n_sample}")
            for j in range(i+1, n_sample):
                d = dtw_distance(X_sample[i], X_sample[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        print("KMeans clustering on distance matrix...")
        kmeans = KMeans(n_clusters=min(n_clusters, n_sample//2), 
                       random_state=42, n_init=10)
        labels_sample = kmeans.fit_predict(dist_matrix)
        
        # Extend labels to the whole dataset
        labels = np.zeros(X.shape[0], dtype=int)
        labels[:n_sample] = labels_sample

        sil = silhouette_score(dist_matrix, labels_sample, metric='precomputed')
        dbi = davies_bouldin_score(dist_matrix, labels_sample)
        chi = calinski_harabasz_score(dist_matrix, labels_sample)

        metrics = {'sil': sil, 'dbi': dbi, 'chi': chi}

        cluster_sizes = dict(zip(*np.unique(labels[:n_sample], return_counts=True)))
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Calinski-Harabasz Index: {chi:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")

        return labels, metrics
    except Exception as e:
        print(f"[ERROR] DTW clustering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# APPROACH 3: sktime TimeSeriesKMeans
# ============================================================================

def cluster_sktime(X, n_clusters=4):
    """
    sktime: built-in TimeSeriesKMeans with DTW metric
    """
    if not SKTIME_AVAILABLE:
        print("[SKIP] sktime not available")
        return None, None

    print("\n" + "="*80)
    print("APPROACH 3: sktime TimeSeriesKMeans (via separate environment)")
    print("="*80)

    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_input = f.name
            pickle.dump({'X': X, 'n_clusters': n_clusters}, f)
        
        sktime_script = f"""
import pickle
import numpy as np
from sktime.clustering.k_means import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

with open('{temp_input}', 'rb') as f:
    data = pickle.load(f)
    
X = data['X']
n_clusters = min(data['n_clusters'], X.shape[0]//2)

X_scaled = np.array([((x - x.mean()) / (x.std() + 1e-8)) for x in X])

kmeans = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    random_state=42,
    n_init=10
)
labels = kmeans.fit_predict(X_scaled)

X_flat = X.reshape(X.shape[0], -1)
sil = silhouette_score(X_flat, labels)
dbi = davies_bouldin_score(X_flat, labels)
chi = calinski_harabasz_score(X_flat, labels)

results = {{
    'labels': labels,
    'metrics': {{'sil': sil, 'dbi': dbi, 'chi': chi}}
}}

with open('{temp_input}.out', 'wb') as f:
    pickle.dump(results, f)
"""
        
        result = subprocess.run(
            [SKTIME_VENV_PATH, '-c', sktime_script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"[ERROR] sktime script error: {result.stderr}")
            return None, None
        
        with open(f'{temp_input}.out', 'rb') as f:
            results = pickle.load(f)
        
        labels = results['labels']
        metrics = results['metrics']
        
        cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Silhouette Score: {metrics['sil']:.4f}")
        print(f"Davies-Bouldin Index: {metrics['dbi']:.4f}")
        print(f"Calinski-Harabasz Index: {metrics['chi']:.2f}")
        print(f"Cluster sizes: {cluster_sizes}")
        
        import os
        os.unlink(temp_input)
        os.unlink(f'{temp_input}.out')
        
        return labels, metrics
        
    except Exception as e:
        print(f"[ERROR] sktime clustering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# GLOBAL MODEL: CatBoostMultiSegmentModel (TRUE CROSS-LEARNING)
# ============================================================================

def build_global_model(X_train, horizon=24):
    """
    GLOBAL MODEL:
    CatBoostMultiSegmentModel trains on ALL segments simultaneously.
    """
    if not ETNA_AVAILABLE:
        print("[ERROR] ETNA not installed!")
        return {}

    try:
        print("\n[GLOBAL MODEL] Building CatBoostMultiSegmentModel...")
        df_list = []
        timestamps = pd.date_range('2020-01-01', periods=X_train.shape[1], freq='H')

        for meter_id in range(X_train.shape[0]):
            df = pd.DataFrame({
                'timestamp': timestamps,
                'segment': f'meter_{meter_id}',
                'target': X_train[meter_id],
            })
            df_list.append(df)
        
        print(f"  -> Preparing ETNA TSDataset: {X_train.shape[0]} segments, {X_train.shape[1]} points")
        
        df_long = pd.concat(df_list, ignore_index=True)
        df_wide = TSDataset.to_dataset(df_long)
        ts_dataset = TSDataset(df=df_wide, freq='H')

        # Advanced feature engineering for peak tracking
        transforms = []
        if LogTransform:
            transforms.append(LogTransform(in_column='target'))
            
        transforms.extend([
            LagTransform(in_column='target', lags=[1, 2, 3, 24, 48]),
            DateFlagsTransform(day_number_in_week=True, is_weekend=True),
            StandardScalerTransform(in_column='target'),
        ])
        
        if StatisticsTransform:
            # Check if it is MeanTransform or StatisticsTransform base
            try:
                # MeanTransform usually just takes window
                transforms.append(StatisticsTransform(in_column='target', window=12, out_column='rolling_mean_12'))
            except:
                pass
        
        print(f"  [DEBUG] X_train characteristics: min={X_train.min():.2f}, max={X_train.max():.2f}, mean={X_train.mean():.2f}")

        # model = CatBoostMultiSegmentModel(
        #     iterations=500,
        #     depth=6,
        #     learning_rate=0.03,
        #     random_seed=42,
        #     # No verbosity settings due to ETNA conflicts
        # )
        model = CatBoostMultiSegmentModel(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
        )

        pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
        print("  -> Training pipeline...")
        pipeline.fit(ts_dataset)
        
        forecast_ts = pipeline.forecast()
        forecast_df = forecast_ts.to_pandas()
        
        print(f"  -> Forecast received. Shape: {forecast_df.shape}")
        print(f"  -> Columns: {forecast_df.columns.names}")
        print(f"  -> Index: {forecast_df.index.name}")
        
        forecasts = {}
        # Get all first-level columns (segments)
        try:
            segments = forecast_df.columns.get_level_values(0).unique()
        except:
            segments = []

        for segment in segments:
            try:
                # Attempt 1: [segment]['target']
                if 'target' in forecast_df[segment].columns:
                    data = forecast_df[segment]['target'].values
                # Attempt 2: [segment][0] (first segment column)
                else:
                    data = forecast_df[segment].iloc[:, 0].values
                
                # Check for NaNs and length
                if len(data) == horizon and not np.isnan(data).all():
                    forecasts[segment] = data
            except:
                pass
        
        print(f"  + Extracted {len(forecasts)}/{len(segments)} segments")
        
        if not forecasts:
            print("[WARNING] Forecast data not found! Columns:")
            print(forecast_df.columns.tolist()[:5])
            print("First rows:")
            print(forecast_df.head())
            
        return forecasts

    except Exception as e:
        print(f"[ERROR] Critical error in build_global_model: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ============================================================================
# CLUSTER MODELS: Forecasting per individual cluster
# ============================================================================

def build_cluster_models(X_train, cluster_labels, horizon=24):
    """
    For each cluster:
    1. Select series belonging to the cluster
    2. Build CatBoostMultiSegmentModel on all these series together
    3. Save forecasts BY SEGMENT_KEY
    """
    if not ETNA_AVAILABLE:
        print("[ERROR] ETNA not installed!")
        return {}

    print("\n[CLUSTER MODELS] Building CatBoostMultiSegmentModel for each cluster...")
    
    all_cluster_forecasts = {}
    
    try:
        for cluster_id in np.unique(cluster_labels):
            print(f"\n  [Cluster {cluster_id}] Building model...")
            
            cluster_mask = cluster_labels == cluster_id
            cluster_meter_ids = np.where(cluster_mask)[0]
            
            if len(cluster_meter_ids) == 0:
                continue
            
            df_list = []
            timestamps = pd.date_range('2020-01-01', periods=X_train.shape[1], freq='H')
            
            for meter_id in cluster_meter_ids:
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'segment': f'meter_{meter_id}',  # segment_key = meter_{meter_id}
                    'target': X_train[meter_id],
                })
                df_list.append(df)
            
            print(f"    -> {len(cluster_meter_ids)} segments in cluster")
            
            df_long = pd.concat(df_list, ignore_index=True)
            df_wide = TSDataset.to_dataset(df_long)
            ts_dataset = TSDataset(df=df_wide, freq='H')
            
            # Match global configuration with Log and Rolling stats
            transforms = []
            if LogTransform:
                transforms.append(LogTransform(in_column='target'))
                
            transforms.extend([
                LagTransform(in_column='target', lags=[1, 2, 3, 24, 48]),
                DateFlagsTransform(day_number_in_week=True, is_weekend=True),
                StandardScalerTransform(in_column='target'),
            ])
            
            if StatisticsTransform:
                try:
                    transforms.append(StatisticsTransform(in_column='target', window=12, out_column='rolling_mean_12'))
                except:
                    pass
            
            model = CatBoostMultiSegmentModel(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                random_seed=42,
            )

            pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
            pipeline.fit(ts_dataset)
            
            forecast_ts = pipeline.forecast()
            forecast_df = forecast_ts.to_pandas()
            
            cluster_forecasts = {}
            # Extract segments
            try:
                segments = forecast_df.columns.get_level_values(0).unique()
            except:
                segments = []

            for segment in segments:
                try:
                    # Attempt 1: [segment]['target']
                    if 'target' in forecast_df[segment].columns:
                        data = forecast_df[segment]['target'].values
                    # Attempt 2: [segment][0]
                    else:
                        data = forecast_df[segment].iloc[:, 0].values
                    
                    if len(data) == horizon and not np.isnan(data).all():
                        cluster_forecasts[segment] = data
                except:
                    pass
            
            all_cluster_forecasts[cluster_id] = cluster_forecasts
            print(f"    + Generated {len(cluster_forecasts)}/{len(segments)} forecasts for cluster {cluster_id}")
        
        return all_cluster_forecasts

    except Exception as e:
        print(f"[ERROR] Cluster models building: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ============================================================================
# ENSEMBLE AGGREGATION
# ============================================================================

def aggregate_cluster_forecasts(cluster_labels, cluster_forecasts):
    """
    PROPER AGGREGATION:
    For each meter_i:
    1. Find its cluster_id from cluster_labels[i]
    2. Get forecast FROM THAT cluster for segment_key = f'meter_{i}'
    3. Assemble the ensemble
    """
    print("\n[ENSEMBLE AGGREGATION]")
    
    ensemble = {}
    n_segments = len(cluster_labels)
    
    for meter_id in range(n_segments):
        cluster_id = cluster_labels[meter_id]
        segment_key = f'meter_{meter_id}'
        
        if cluster_id in cluster_forecasts:
            # Match by segment_key
            if segment_key in cluster_forecasts[cluster_id]:
                ensemble[segment_key] = cluster_forecasts[cluster_id][segment_key]
            else:
                print(f"  [!] {segment_key} not found in cluster {cluster_id}")
    
    print(f"Aggregated {len(ensemble)}/{n_segments} segments into ensemble")
    return ensemble

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_forecasts(X_val, global_forecasts, ensemble_forecasts, horizon):
    """Compare global model and ensemble"""
    print("\n" + "="*80)
    print("EVALUATION: Global Model vs Cluster-Ensemble")
    print("="*80)
    
    if not global_forecasts:
        print("[DEBUG] global_forecasts empty in evaluate_forecasts!")
    if not ensemble_forecasts:
        print("[DEBUG] ensemble_forecasts empty in evaluate_forecasts!")

    results = {
        'global': {},
        'ensemble': {},
        'global_errors': [],
        'ensemble_errors': [],
        'per_meter': {} # To store {meter_num: {'global_mae': ..., 'ensemble_mae': ...}}
    }

    # Global evaluation
    global_maes = []
    global_rmses = []
    global_mapes = []
    global_errors = []
    
    per_meter_global = {}

    if global_forecasts:
        for segment_key, forecast in global_forecasts.items():
            try:
                # Safe meter number extraction
                meter_num = int(segment_key.split('_')[-1])
                
                if meter_num < 0 or meter_num >= X_val.shape[0]:
                    continue
                    
                actual = X_val[meter_num][:len(forecast)]
                
                # Skip if forecast contains NaNs or invalid length
                if np.isnan(forecast).any() or len(forecast) == 0:
                    continue
                
                errors = actual - forecast
                global_errors.extend(errors.tolist())
                
                mae = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mape = np.mean(np.abs((actual - forecast) / (np.abs(actual) + 1e-8))) * 100

                global_maes.append(mae)
                global_rmses.append(rmse)
                global_mapes.append(mape)
                
                per_meter_global[meter_num] = mae
            except Exception as e:
                if len(global_maes) < 5:
                    print(f"  [DEBUG] Segment evaluation {segment_key} failed: {e}")
                pass
        
        results['global'] = {
            'MAE': np.mean(global_maes),
            'RMSE': np.mean(global_rmses),
            'MAPE': np.mean(global_mapes)
        }
        results['global_errors'] = global_errors
        
        print(f"\n[GLOBAL CatBoost Model]")
        print(f"  MAE:  {results['global']['MAE']:.4f}")
        print(f"  RMSE: {results['global']['RMSE']:.4f}")
        print(f"  MAPE: {results['global']['MAPE']:.4f}")

    # Ensemble evaluation
    ensemble_maes = []
    ensemble_rmses = []
    ensemble_mapes = []
    ensemble_errors = []

    if ensemble_forecasts:
        for segment_key, forecast in ensemble_forecasts.items():
            try:
                meter_num = int(segment_key.split('_')[1])
                actual = X_val[meter_num][:len(forecast)]
                
                errors = actual - forecast
                ensemble_errors.extend(errors.tolist())
                
                mae = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mape = np.mean(np.abs((actual - forecast) / (np.abs(actual) + 1e-8))) * 100
                
                ensemble_maes.append(mae)
                ensemble_rmses.append(rmse)
                ensemble_mapes.append(mape)
                
                if meter_num in per_meter_global:
                    results['per_meter'][meter_num] = {
                        'global_mae': per_meter_global[meter_num],
                        'ensemble_mae': mae,
                        'improvement': (per_meter_global[meter_num] - mae) / (per_meter_global[meter_num] + 1e-8)
                    }
            except:
                pass
        
        results['ensemble'] = {
            'MAE': np.mean(ensemble_maes),
            'RMSE': np.mean(ensemble_rmses),
            'MAPE': np.mean(ensemble_mapes)
        }
        results['ensemble_errors'] = ensemble_errors
        
        print(f"\n[CLUSTER-ENSEMBLE CatBoost Model]")
        print(f"  MAE:  {results['ensemble']['MAE']:.4f}")
        print(f"  RMSE: {results['ensemble']['RMSE']:.4f}")
        print(f"  MAPE: {results['ensemble']['MAPE']:.4f}")

    return results

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def diebold_mariano_test(errors1, errors2, h=1):
    """DM test for comparing forecast accuracy"""
    d = np.array(errors1)**2 - np.array(errors2)**2
    n = len(d)
    
    mean_d = np.mean(d)
    
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        if len(d) > k:
            gamma_k = np.cov(d[:-k], d[k:])[0, 1]
            gamma_sum += gamma_k
    
    var_d = gamma_0 + 2 * gamma_sum
    var_d = max(var_d, 1e-10)
    
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def run_statistical_tests(global_errors, ensemble_errors, horizon=24):
    """Statistical tests"""
    print("\n" + "="*80)
    print("STATISTICAL TESTS: Global vs Cluster-Ensemble")
    print("="*80)
    
    results = {}
    
    dm_stat, dm_pval = diebold_mariano_test(global_errors, ensemble_errors, h=horizon)
    results['diebold_mariano'] = {'statistic': dm_stat, 'p_value': dm_pval}
    
    print(f"\n[Diebold-Mariano Test]")
    print(f"  H0: Both models have same accuracy")
    print(f"  DM Statistic: {dm_stat:.4f}")
    print(f"  p-value: {dm_pval:.4f}")
    
    if dm_pval < 0.05:
        if dm_stat > 0:
            print(f"  -> ENSEMBLE is significantly BETTER (p < 0.05)")
        else:
            print(f"  -> GLOBAL is significantly BETTER (p < 0.05)")
    else:
        print(f"  -> No significant difference (p >= 0.05)")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cluster_visualization(X, labels, approach_name):
    """Visualize clusters"""
    print(f"\n[PLOT] Visualizing clusters {approach_name}...")
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    fig, axes = plt.subplots(min(n_clusters, 3), 1, figsize=(15, 5*min(n_clusters, 3)), sharex=True)
    if n_clusters == 1: axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, label in enumerate(unique_labels[:3]):
        cluster_data = X[labels == label]
        
        if len(cluster_data) == 0: continue
            
        centroid = np.mean(cluster_data, axis=0)
        
        limit = min(len(cluster_data), 50)
        for j in range(limit):
            axes[i].plot(cluster_data[j], color=colors[i % 10], alpha=0.15) 
            
        axes[i].plot(centroid, color=colors[i % 10], linewidth=3, label='Centroid')
        axes[i].set_title(f'Cluster {label} ({len(cluster_data)} time series)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.2)
        
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f'{approach_name.lower().replace(" ", "_")}_clusters.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {filename}")

def get_best_representative_meters(per_meter_results, top_n=4):
    """
    Select meters where the ensemble model performs significantly better 
    than the global model, or at least shows the best improvement.
    """
    if not per_meter_results:
        return []
    
    # Sort by improvement (descending)
    sorted_meters = sorted(
        per_meter_results.items(), 
        key=lambda x: x[1]['improvement'], 
        reverse=True
    )
    
    return [m[0] for m in sorted_meters[:top_n]]

def plot_forecasts_comparison(X_val, global_forecasts, ensemble_forecasts, approach_name, horizon, best_meters=None):
    """Compare forecasts with improved aesthetics and multi-panel support"""
    print(f"\n[PLOT] Comparing forecasts {approach_name}...")
    
    if best_meters is None or len(best_meters) == 0:
        # Fallback to a single sample meter if none provided
        best_meters = [min(10, X_val.shape[0] - 1)]
    
    n_meters = len(best_meters)
    cols = 2
    rows = (n_meters + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), squeeze=False)
    axes = axes.flatten()

    sns.set_style("whitegrid")
    palette = sns.color_palette("muted")
    
    for i, meter_id in enumerate(best_meters):
        ax = axes[i]
        actual = X_val[meter_id][:horizon]
        
        global_fc_key = f'meter_{meter_id}'
        global_fc = global_forecasts.get(global_fc_key, actual) if global_forecasts else actual
        ensemble_fc = ensemble_forecasts.get(global_fc_key, actual) if ensemble_forecasts else actual

        mae_global = mean_absolute_error(actual, global_fc) if not np.isnan(global_fc).all() else np.nan
        mae_ensemble = mean_absolute_error(actual, ensemble_fc) if not np.isnan(ensemble_fc).all() else np.nan

        ax.plot(actual, color='black', linewidth=2, marker='o', markersize=4, label='Actual', alpha=0.7, zorder=3)
        ax.plot(global_fc, color=palette[0], linestyle='-', linewidth=2, 
                label=f'Global (MAE: {mae_global:.3f})', alpha=0.9, zorder=2)
        ax.plot(ensemble_fc, color=palette[1], linestyle='--', linewidth=2, 
                label=f'Ensemble (MAE: {mae_ensemble:.3f})', alpha=0.9, zorder=2)

        ax.set_title(f'Meter: {meter_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Consumption', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, horizon - 0.5)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Forecast Comparison: {approach_name} Approach\n(Selected Best-Performing Meters for Ensemble)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filename = os.path.join(PLOTS_DIR, f'{approach_name.lower().replace(" ", "_")}_forecast_comparison.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  + Saved: {filename}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """PIPELINE with Backtesting"""
    print("\n" + "="*80)
    print("PIPELINE: Backtesting + Multi-Segment Forecasting")
    print("="*80)

    # Load data
    try:
        X = load_powercons_data()
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}. Falling back to synthetic.")
        X = load_synthetic_data()
    
    horizon = 24
    n_clusters = 4
    
    # 6 backtesting windows for strong statistical power (n=144 samples)
    # Each window: 18 points train, then 18 points test
    n_total = X.shape[1]
    window_size = int(n_total * 0.075)  # 18 timesteps per test window
    
    test_windows = [
        (int(0.3*n_total), int(0.375*n_total)),
        (int(0.375*n_total), int(0.45*n_total)),
        (int(0.45*n_total), int(0.525*n_total)),
        (int(0.525*n_total), int(0.6*n_total)),
        (int(0.6*n_total), int(0.675*n_total)),
        (int(0.675*n_total), int(0.75*n_total)),
    ]

    print(f"\n[BACKTESTING] Starting evaluation on {len(test_windows)} windows...")

    # Test each clustering approach
    approaches = []
    labels_tsfresh, _ = cluster_tsfresh(X, n_clusters=n_clusters)
    if labels_tsfresh is not None: approaches.append(('TSFresh', labels_tsfresh))
    
    labels_dtw, _ = cluster_dtw(X, n_clusters=n_clusters)
    if labels_dtw is not None: approaches.append(('DTW', labels_dtw))
    
    labels_sktime, _ = cluster_sktime(X, n_clusters=n_clusters)
    if labels_sktime is not None: approaches.append(('sktime', labels_sktime))

    for approach_name, labels in approaches:
        print(f"\n{'='*80}")
        print(f"PIPELINE for {approach_name} (Backtesting Mode)")
        print(f"{'='*80}")
        
        all_global_errors = []
        all_ensemble_errors = []
        
        final_global_forecasts = {}
        final_ensemble_forecasts = {}
        final_X_val = None

        for i, (train_end, test_end) in enumerate(test_windows):
            print(f"\n--- Backtest Window {i+1} (Train end: {train_end}, Test: {train_end}:{test_end}) ---")
            
            X_train = X[:, :train_end]
            X_val = X[:, train_end:test_end]
            
            current_horizon = test_end - train_end
            
            # Step 1: Global Model
            global_forecasts = build_global_model(X_train, horizon=current_horizon)
            
            # Step 2: Cluster Models
            cluster_forecasts = build_cluster_models(X_train, labels, horizon=current_horizon)
            
            # Step 3: Aggregate
            ensemble_forecasts = aggregate_cluster_forecasts(labels, cluster_forecasts)
            
            # Step 4: Evaluate this window
            eval_results = evaluate_forecasts(X_val, global_forecasts, ensemble_forecasts, current_horizon)
            
            # Accumulate errors for collective statistical test
            all_global_errors.extend(eval_results.get('global_errors', []))
            all_ensemble_errors.extend(eval_results.get('ensemble_errors', []))
            
            # Keep the last window for visualization
            final_global_forecasts = global_forecasts
            final_ensemble_forecasts = ensemble_forecasts
            final_X_val = X_val

        
        print(f"\n[FINAL SUMMARY - {approach_name}]")
        if all_global_errors and all_ensemble_errors:
            stat_results = run_statistical_tests(
                all_global_errors,
                all_ensemble_errors,
                horizon=horizon
            )
            
            global_mae = np.mean(np.abs(all_global_errors))
            ensemble_mae = np.mean(np.abs(all_ensemble_errors))
            print(f"Overall Global MAE:   {global_mae:.4f}")
            print(f"Overall Ensemble MAE: {ensemble_mae:.4f}")
            print(f"Improvement:         {((global_mae - ensemble_mae)/global_mae)*100:.2f}%")

        # Visualization of the last window
        plot_cluster_visualization(X, labels, approach_name)
        
        # Select best meters for plotting
        if 'per_meter' in eval_results:
            best_meters = get_best_representative_meters(eval_results['per_meter'], top_n=4)
        else:
            best_meters = None
            
        plot_forecasts_comparison(final_X_val, final_global_forecasts, final_ensemble_forecasts, approach_name, horizon, best_meters=best_meters)
    
    print("\n" + "="*80)
    print("PIPELINE FINISHED")
    print("="*80)

if __name__ == '__main__':
    main()