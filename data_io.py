"""
Dataset loaders for the DAMDID 2026 clustered-local forecasting study.

Two open daily panels:
    ELEC --- Worldwide Electricity Load (Mendeley ybggkc58fz)
    NN5  --- ATM cash withdrawals (Monash TSF Repository)

Each loader returns a tuple (X, names, start_date_str) where:
    X        ndarray of shape (n_series, n_timesteps), float64
    names    list of length n_series with human-readable identifiers
    start_date_str  ISO calendar date of the first column of X
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# ELEC -- Worldwide Electricity Load (Mendeley)
# ---------------------------------------------------------------------------

# Resolve dataset locations against the project root, not the cwd, so the
# pipeline works whether invoked from `code/` or from the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _project_path(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, "data", *parts)


ELEC_BASE_PATH = _project_path("Worldwide Electricity Load Dataset", "GloElecLoad")

ELEC_WINDOW_START = pd.Timestamp("2022-01-01")
ELEC_WINDOW_END = pd.Timestamp("2022-12-31")
ELEC_COVERAGE_THRESHOLD = 0.70
ELEC_MIN_OBS = 200


def _detect_datetime_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        c = str(col).lower()
        if any(tok in c for tok in ("time", "date", "datetime", "timestamp")):
            return col
    return df.columns[0]


def _detect_target_col(df: pd.DataFrame, dt_col: str) -> Optional[str]:
    for col in df.columns:
        if col == dt_col:
            continue
        c = str(col).lower()
        if any(tok in c for tok in ("load", "mw", "demand", "value", "consumption")):
            return col
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c != dt_col]
    return numeric[0] if numeric else None


def _parse_elec_csv(path: str) -> Optional[pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    if df.empty or df.shape[1] < 2:
        return None
    dt_col = _detect_datetime_col(df)
    target_col = _detect_target_col(df, dt_col)
    if target_col is None:
        return None
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=[dt_col, target_col])
    if df.empty:
        return None
    df = df[[dt_col, target_col]].set_index(dt_col).sort_index()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    daily = df[target_col].resample("D").mean()
    daily = daily.where(daily > 0, np.nan)
    if daily.notna().sum() < 100:
        return None
    return daily


def load_elec(base_path: str = ELEC_BASE_PATH) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load and align the Mendeley Worldwide Electricity Load panel."""
    if not os.path.exists(base_path):
        return None, None, None

    all_series: dict[str, pd.Series] = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".csv") or "Additional_Information" in file or file.startswith("."):
                continue
            path = os.path.join(root, file)
            region = os.path.basename(root)
            if region == "GloElecLoad":
                region = file.replace(".csv", "")
            try:
                series = _parse_elec_csv(path)
                if series is None:
                    continue
                key = region.strip()
                if key in all_series:
                    key = f"{key}_{file.replace('.csv', '')}"
                all_series[key] = series
            except Exception:
                continue

    if not all_series:
        return None, None, None

    combined = pd.DataFrame(all_series)
    idx = pd.date_range(ELEC_WINDOW_START, ELEC_WINDOW_END, freq="D")
    combined = combined.reindex(idx)

    coverage = combined.notna().mean()
    valid_cols = coverage[coverage >= ELEC_COVERAGE_THRESHOLD].index.tolist()
    if len(valid_cols) < 8:
        valid_cols = coverage.sort_values(ascending=False).head(min(30, len(coverage))).index.tolist()

    combined = combined[valid_cols]
    combined = combined.loc[:, combined.notna().sum() >= ELEC_MIN_OBS]
    combined = combined.interpolate(method="linear", limit_direction="both").ffill().bfill()
    combined = combined.loc[:, combined.std(axis=0) > 1e-8]

    if combined.shape[1] < 4:
        return None, None, None

    X = combined.T.values.astype(float)
    return X, combined.columns.tolist(), ELEC_WINDOW_START.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# NN5 -- ATM cash withdrawals (Monash TSF Repository)
# ---------------------------------------------------------------------------

# Monash hosts the canonical bundle on Zenodo.  The "with missing values"
# variant preserves the original NaN pattern; we impute below to match
# the ELEC preprocessing path.  Source DOI: 10.5281/zenodo.4656110.
NN5_DOWNLOAD_URL = (
    "https://zenodo.org/records/4656110/files/nn5_daily_dataset_with_missing_values.zip"
)
NN5_LOCAL_DIR = _project_path("nn5")
NN5_TSF_FILENAME = "nn5_daily_dataset_with_missing_values.tsf"


def _download_nn5(target_dir: str = NN5_LOCAL_DIR) -> str:
    """Download the NN5 .tsf bundle if not already cached. Return file path."""
    os.makedirs(target_dir, exist_ok=True)
    tsf_path = os.path.join(target_dir, NN5_TSF_FILENAME)
    if os.path.exists(tsf_path):
        return tsf_path
    print(f"  Downloading NN5 bundle from {NN5_DOWNLOAD_URL} ...")
    response = requests.get(NN5_DOWNLOAD_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(target_dir)
    if not os.path.exists(tsf_path):
        # Some archive variants nest the file deeper.
        for root, _, files in os.walk(target_dir):
            for f in files:
                if f == NN5_TSF_FILENAME:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"NN5 .tsf not found after extraction in {target_dir}")
    return tsf_path


def _parse_tsf(path: str) -> Tuple[List[str], List[pd.Timestamp], List[np.ndarray]]:
    """
    Minimal TSF parser sufficient for Monash daily benchmarks.

    Returns parallel lists of: series names, start timestamps, value arrays
    (NaNs preserved for ``? ``-encoded missing entries).
    """
    names: List[str] = []
    starts: List[pd.Timestamp] = []
    values: List[np.ndarray] = []
    in_data = False
    attribute_names: List[str] = []
    # Stream-read the file line by line.  Earlier versions slurped
    # the whole .tsf into memory which is fine for small Monash
    # bundles (NN5: 1.4 MB) but fails for large ones (Weather: 200+ MB)
    # by combining a 600 MB+ Python string with downstream ETNA init.
    fh = open(path, "rb")
    for raw_b in fh:
        raw = raw_b.decode("utf-8", errors="replace")
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@attribute"):
            parts = line.split()
            attribute_names.append(parts[1])
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue

        # Data line layout: attr1:attr2:...:v1,v2,...,vN
        head_parts = line.split(":")
        if len(head_parts) < len(attribute_names) + 1:
            continue
        attr_values = head_parts[: len(attribute_names)]
        value_field = ":".join(head_parts[len(attribute_names):])

        name = attr_values[0] if attribute_names else ""
        start_ts = pd.NaT
        for attr_name, attr_val in zip(attribute_names, attr_values):
            if attr_name == "series_name":
                name = attr_val
            elif attr_name == "start_timestamp":
                try:
                    start_ts = pd.to_datetime(attr_val.split("_")[0])
                except Exception:
                    start_ts = pd.NaT

        # Use explicit for-loop instead of list comprehension to avoid
        # an intermittent CPython internal error observed on this path.
        tokens = value_field.split(",")
        arr = np.empty(len(tokens), dtype=float)
        for i, t in enumerate(tokens):
            t = t.strip()
            if not t or t == "?":
                arr[i] = np.nan
            else:
                try:
                    arr[i] = float(t)
                except ValueError:
                    arr[i] = np.nan
        names.append(name)
        starts.append(start_ts if not pd.isna(start_ts) else pd.Timestamp("1996-03-18"))
        values.append(arr)
    fh.close()
    return names, starts, values


def load_nn5(local_dir: str = NN5_LOCAL_DIR) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load the Monash NN5 daily ATM cash withdrawal panel."""
    try:
        tsf_path = _download_nn5(local_dir)
    except Exception as exc:
        print(f"  [!] NN5 download failed: {exc}")
        return None, None, None

    names, starts, values = _parse_tsf(tsf_path)
    if not values:
        return None, None, None

    # All NN5 series share a common start timestamp and equal length, but we
    # align to the latest common window defensively.
    common_start = max(starts)
    aligned: List[np.ndarray] = []
    for arr, start in zip(values, starts):
        offset = (common_start - start).days
        offset = max(offset, 0)
        aligned.append(arr[offset:])
    min_len = min(len(a) for a in aligned)
    matrix = np.vstack([a[:min_len] for a in aligned]).astype(float)

    # Impute missing values per series (linear interp + edge fill).
    df = pd.DataFrame(matrix.T)
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    matrix = df.T.values

    # Drop zero-variance series defensively.
    stds = matrix.std(axis=1)
    keep = stds > 1e-8
    matrix = matrix[keep]
    kept_names = [n for n, k in zip(names, keep) if k]
    return matrix, kept_names, common_start.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# M4 Daily -- a heterogeneous M-competition daily panel (Monash mirror)
# ---------------------------------------------------------------------------

# Monash hosts the canonical M4 daily bundle on Zenodo as a .tsf archive.
# The full dataset has 4,227 daily series of varying length (~14 to ~9,933
# days, median ~2,400) drawn from M4's six categories (Demographic,
# Finance, Industry, Macro, Micro, Other).  We subset to a panel large
# enough to exercise the pipeline but small enough to keep wall-clock
# under one hour per (horizon, method): ``M4D_TARGET_N`` longest series,
# truncated to the last ``M4D_TRUNCATE_LEN`` observations and aligned
# to a synthetic common start date.  M4 series do not share a real
# calendar so day-of-week features are intentionally synthetic; this
# is a known limitation accepted by the M4 organisers themselves and
# does not affect the per-series MAE comparison.
M4D_DOWNLOAD_URL = "https://zenodo.org/records/4656548/files/m4_daily_dataset.zip"
M4D_LOCAL_DIR = _project_path("m4_daily")
M4D_TSF_FILENAME = "m4_daily_dataset.tsf"
M4D_MIN_LENGTH = 1200
M4D_TARGET_N = 80
M4D_TRUNCATE_LEN = 1200
M4D_SEED = 42


def _download_m4_daily(target_dir: str = M4D_LOCAL_DIR) -> str:
    os.makedirs(target_dir, exist_ok=True)
    tsf_path = os.path.join(target_dir, M4D_TSF_FILENAME)
    if os.path.exists(tsf_path):
        return tsf_path
    print(f"  Downloading M4 daily bundle from {M4D_DOWNLOAD_URL} ...")
    response = requests.get(M4D_DOWNLOAD_URL, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(target_dir)
    if not os.path.exists(tsf_path):
        for root, _, files in os.walk(target_dir):
            for f in files:
                if f == M4D_TSF_FILENAME:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"M4 daily .tsf not found in {target_dir}")
    return tsf_path


def load_m4_daily(local_dir: str = M4D_LOCAL_DIR) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load a balanced subset of the Monash M4 daily panel.

    Returns the matrix ``(N, T)``, the kept series names, and the
    synthetic common start date used when constructing the matrix.
    """
    try:
        tsf_path = _download_m4_daily(local_dir)
    except Exception as exc:
        print(f"  [!] M4 daily download failed: {exc}")
        return None, None, None

    names, _, values = _parse_tsf(tsf_path)
    if not values:
        return None, None, None

    long_enough = [(n, v) for n, v in zip(names, values) if len(v) >= M4D_MIN_LENGTH]
    if len(long_enough) < 4:
        print(f"  [!] M4 daily has only {len(long_enough)} long series")
        return None, None, None

    if len(long_enough) > M4D_TARGET_N:
        rng = np.random.default_rng(M4D_SEED)
        idx = rng.choice(len(long_enough), size=M4D_TARGET_N, replace=False)
        sampled = [long_enough[i] for i in sorted(idx)]
    else:
        sampled = long_enough

    aligned = []
    for n, v in sampled:
        v_trunc = v[-M4D_TRUNCATE_LEN:]
        aligned.append((n, v_trunc))

    matrix = np.vstack([v for _, v in aligned]).astype(float)

    # Impute the small number of NaNs preserved by the TSF parser.
    df = pd.DataFrame(matrix.T)
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    matrix = df.T.values

    # Drop pathological constant series defensively (e.g. all-zero stretches).
    stds = matrix.std(axis=1)
    keep = stds > 1e-8
    matrix = matrix[keep]
    kept_names = [aligned[i][0] for i in range(len(aligned)) if keep[i]]

    # Use a synthetic common start date; see module docstring.
    common_start = pd.Timestamp("2010-01-01")
    return matrix, kept_names, common_start.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# WEATHER -- Australian daily weather (Monash mirror)
# ---------------------------------------------------------------------------

# Monash weather dataset: thousands of Australian weather stations with
# daily readings going back to early 20th century.  We deliberately
# subset to a small slice ($N{=}60$ stations, last $T{=}1{,}000$ days)
# to keep wall-clock under one hour per (horizon) cell on our
# evaluation host, and to focus the heterogeneity story on the
# climate-zone shape variation that DTW clustering targets.  All
# stations share a real calendar (the dataset's reference end date),
# so day-of-week and weekday flags carry their conventional semantics.
WEATHER_DOWNLOAD_URL = "https://zenodo.org/records/4654822/files/weather_dataset.zip"
WEATHER_LOCAL_DIR = _project_path("weather")
WEATHER_TSF_FILENAME = "weather_dataset.tsf"
WEATHER_TARGET_N = 60
WEATHER_TRUNCATE_LEN = 1000
WEATHER_SEED = 42


def _download_weather(target_dir: str = WEATHER_LOCAL_DIR) -> str:
    os.makedirs(target_dir, exist_ok=True)
    tsf_path = os.path.join(target_dir, WEATHER_TSF_FILENAME)
    if os.path.exists(tsf_path):
        return tsf_path
    print(f"  Downloading Weather bundle from {WEATHER_DOWNLOAD_URL} ...")
    response = requests.get(WEATHER_DOWNLOAD_URL, timeout=300)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(target_dir)
    if not os.path.exists(tsf_path):
        for root, _, files in os.walk(target_dir):
            for f in files:
                if f == WEATHER_TSF_FILENAME:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"weather .tsf not found in {target_dir}")
    return tsf_path


def load_weather(local_dir: str = WEATHER_LOCAL_DIR) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load a small subset of the Monash weather panel.

    The Weather .tsf is large (~200 MB).  We stream-parse it and stop
    early after collecting ``WEATHER_TARGET_N * 4`` series that meet
    the minimum-length contract, then subsample uniformly to the
    target $N$.  This keeps peak memory under a few hundred MB even
    when the parser shares process state with ETNA imports.
    """
    try:
        tsf_path = _download_weather(local_dir)
    except Exception as exc:
        print(f"  [!] Weather download failed: {exc}")
        return None, None, None

    target_pool = WEATHER_TARGET_N * 4
    long_enough: List[Tuple[str, np.ndarray]] = []

    in_data = False
    attribute_names: List[str] = []
    fh = open(tsf_path, "rb")
    try:
        for raw_b in fh:
            line = raw_b.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@attribute"):
                parts = line.split()
                attribute_names.append(parts[1])
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data or line.startswith("@"):
                continue

            head_parts = line.split(":")
            if len(head_parts) < len(attribute_names) + 1:
                continue
            attr_values = head_parts[: len(attribute_names)]
            value_field = ":".join(head_parts[len(attribute_names):])

            name = attr_values[0] if attribute_names else ""

            tokens = value_field.split(",")
            if len(tokens) < WEATHER_TRUNCATE_LEN:
                continue
            arr = np.empty(len(tokens), dtype=float)
            for i, t in enumerate(tokens):
                t = t.strip()
                if not t or t == "?":
                    arr[i] = np.nan
                else:
                    try:
                        arr[i] = float(t)
                    except ValueError:
                        arr[i] = np.nan
            long_enough.append((name, arr))
            if len(long_enough) >= target_pool:
                break
    finally:
        fh.close()

    if len(long_enough) < 4:
        return None, None, None

    if len(long_enough) > WEATHER_TARGET_N:
        rng = np.random.default_rng(WEATHER_SEED)
        idx = rng.choice(len(long_enough), size=WEATHER_TARGET_N, replace=False)
        sampled = [(long_enough[i][0], None, long_enough[i][1]) for i in sorted(idx)]
    else:
        sampled = [(n, None, v) for n, v in long_enough]

    aligned = []
    for n, s, v in sampled:
        v_trunc = v[-WEATHER_TRUNCATE_LEN:]
        # Push the per-station start forward by the truncation offset.
        offset_days = max(0, len(v) - WEATHER_TRUNCATE_LEN)
        adj_start = s + pd.Timedelta(days=offset_days) if not pd.isna(s) else s
        aligned.append((n, adj_start, v_trunc))

    matrix = np.vstack([v for _, _, v in aligned]).astype(float)

    # Impute small NaN gaps the same way we do for NN5 / ELEC.
    df = pd.DataFrame(matrix.T)
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    matrix = df.T.values

    stds = matrix.std(axis=1)
    keep = stds > 1e-8
    matrix = matrix[keep]
    kept = [aligned[i] for i in range(len(aligned)) if keep[i]]
    kept_names = [n for n, _, _ in kept]

    # Monash weather timestamps are not consistently real-world dates;
    # we use a synthetic common start so DateFlagsTransform behaves
    # consistently across stations.
    common_start = pd.Timestamp("2018-01-01")
    return matrix, kept_names, common_start.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# ROSSMANN -- Kaggle Rossmann Store Sales (third panel, daily)
# ---------------------------------------------------------------------------

# Kaggle Rossmann Store Sales data: 1,115 stores, daily Sales over
# 2013-01-01 to 2015-07-31 (942 days for the well-covered subset).
# Zero values correspond to closed-store days (Open=0); we keep them
# in the time series since they are real and predictable from
# weekday + holiday signals -- consistent with the per-segment
# heterogeneity (different store types and assortments) that DTW
# clustering should pick up.  Subset to ``ROSSMANN_TARGET_N`` stores
# with a full 942-day history to keep memory pressure modest under
# the multi-window pipeline.
ROSSMANN_LOCAL_DIR = _project_path("rossmann")
ROSSMANN_TRAIN_FILE = "train.csv"
ROSSMANN_FULL_LENGTH = 942
ROSSMANN_TARGET_N = 50
ROSSMANN_SEED = 42


def load_rossmann(local_dir: str = ROSSMANN_LOCAL_DIR) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load a balanced subset of the Rossmann daily sales panel."""
    train_path = os.path.join(local_dir, ROSSMANN_TRAIN_FILE)
    if not os.path.exists(train_path):
        print(f"  [!] Rossmann train.csv missing at {train_path}")
        return None, None, None
    df = pd.read_csv(
        train_path,
        usecols=["Store", "Date", "Sales"],
        parse_dates=["Date"],
        dtype={"Store": "int32", "Sales": "int64"},
    )
    df = df.sort_values(["Store", "Date"])
    counts = df.groupby("Store").size()
    full_stores = counts[counts == ROSSMANN_FULL_LENGTH].index.tolist()
    if len(full_stores) < 4:
        return None, None, None

    rng = np.random.default_rng(ROSSMANN_SEED)
    if len(full_stores) > ROSSMANN_TARGET_N:
        chosen = sorted(rng.choice(full_stores, size=ROSSMANN_TARGET_N, replace=False).tolist())
    else:
        chosen = sorted(full_stores)

    sub = df[df["Store"].isin(chosen)]
    pivot = sub.pivot(index="Date", columns="Store", values="Sales")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(columns=chosen)
    pivot = pivot.interpolate(method="linear", limit_direction="both").ffill().bfill()
    matrix = pivot.T.values.astype(float)

    stds = matrix.std(axis=1)
    keep = stds > 1e-8
    matrix = matrix[keep]
    kept_names = [str(s) for s, k in zip(chosen, keep) if k]

    common_start = pivot.index.min().strftime("%Y-%m-%d")
    return matrix, kept_names, common_start


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "ELEC": load_elec,
    "NN5": load_nn5,
    "M4D": lambda: load_m4_daily(),
    "WEATHER": lambda: load_weather(),
    "ROSSMANN": lambda: load_rossmann(),
}


def load_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]()
