"""
Clustering of heterogeneous time series for the clustered-local
forecasting pipeline.

The default algorithm is DTW + Agglomerative Clustering with average
linkage over z-score-normalized series.  Two corrections relative to the
prior code are made here:

* The number of clusters ``k`` is selected by a validation curve over
  the most recent ``h`` days of training data, replacing the previous
  ``min(6, max(3, n // 5))`` heuristic that produced singleton
  clusters on the ELEC panel.
* A minimum cluster size is enforced after the partition is computed.
  Clusters smaller than the threshold are merged into the cluster
  whose centroid is closest in DTW distance, so the downstream model
  fitter never receives a singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

try:
    # Prefer the C-extension implementation: the pure-Python fallback
    # ships an indexing bug on Python >= 3.13.
    from dtaidistance.dtw import distance_fast as _dtw_distance_impl
    from dtaidistance.dtw import distance_matrix_fast as _dtw_matrix_fast
    DTW_AVAILABLE = True
    DTW_MATRIX_FAST_AVAILABLE = True
except ImportError:
    try:
        from dtaidistance.dtw import distance as _dtw_distance_impl  # type: ignore[no-redef]
        DTW_AVAILABLE = True
        DTW_MATRIX_FAST_AVAILABLE = False
        _dtw_matrix_fast = None  # type: ignore[assignment]
    except ImportError:
        DTW_AVAILABLE = False
        DTW_MATRIX_FAST_AVAILABLE = False
        _dtw_matrix_fast = None  # type: ignore[assignment]


def _dtw(a, b):
    """Wrapper that guarantees the input arrays are C-contiguous float64."""
    return _dtw_distance_impl(
        a.astype("float64", copy=False).ravel(),
        b.astype("float64", copy=False).ravel(),
    )


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise z-score: clusters reflect shape, not absolute level."""
    scaler = StandardScaler()
    return scaler.fit_transform(X.T).T


def dtw_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise DTW.

    Switched from the parallel C-extension (``distance_matrix_fast(parallel=True)``)
    to a serial loop over ``distance_fast``: the parallel path interacts
    badly with downstream ETNA / pandas threading on Windows and produces
    intermittent ``Windows fatal exception: access violation`` crashes
    inside ``pandas.concat``.  The serial loop is ~10x slower for the
    DTW step alone but the DTW step is a small fraction of total
    wall-clock (less than one minute on a 111-series panel), and the
    stability gain dominates.
    """
    if not DTW_AVAILABLE:
        raise RuntimeError("dtaidistance is required for DTW clustering")
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _dtw(X[i], X[j])
    return D


# ---------------------------------------------------------------------------
# Partition with minimum cluster size
# ---------------------------------------------------------------------------


def _merge_small_clusters(
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Merge clusters smaller than ``min_size`` into their nearest neighbour.

    Nearest neighbour is defined as the cluster whose member has the
    smallest distance to a member of the small cluster.  This keeps the
    operation deterministic and DTW-consistent.
    """
    labels = labels.copy()
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < min_size]
        if len(small) == 0 or len(unique) <= 1:
            break
        target_label = small[np.argmin(counts[counts < min_size])]
        members = np.where(labels == target_label)[0]
        non_members = np.where(labels != target_label)[0]
        if len(non_members) == 0:
            break
        sub = distance_matrix[np.ix_(members, non_members)]
        nearest_outside = non_members[np.argmin(sub.min(axis=0))]
        labels[members] = labels[nearest_outside]
    # Reindex labels to a contiguous 0..K-1 range.
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse


def cluster_with_min_size(
    distance_matrix: np.ndarray,
    n_clusters: int,
    min_size: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Agglomerative clustering with a KMeans fallback for unbalanced cuts."""
    n = distance_matrix.shape[0]
    n_clusters = max(2, min(n_clusters, n))

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance_matrix)

    counts = np.bincount(labels)
    if counts.max() / counts.sum() > 0.85 and n >= 4:
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(distance_matrix)

    return _merge_small_clusters(labels, distance_matrix, min_size)


# ---------------------------------------------------------------------------
# Validation-MAE curve over k
# ---------------------------------------------------------------------------


@dataclass
class ClusterSelectionResult:
    k: int
    labels: np.ndarray
    val_mae_curve: Dict[int, float] = field(default_factory=dict)


def select_k_by_validation(
    X_train: np.ndarray,
    distance_matrix: np.ndarray,
    holdout_h: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    k_grid: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    min_size: int = 3,
    seed: int = 42,
) -> ClusterSelectionResult:
    """Pick ``k`` that minimizes validation MAE on the last ``holdout_h``
    days.  ``score_fn`` receives ``(X_train_inner, labels)`` and returns
    a scalar MAE.  This indirection keeps the clustering module
    decoupled from the forecasting module.
    """
    if X_train.shape[1] <= holdout_h + 30:
        # Not enough data for a meaningful curve; fall back to k=3.
        labels = cluster_with_min_size(distance_matrix, 3, min_size=min_size, seed=seed)
        return ClusterSelectionResult(k=int(labels.max() + 1), labels=labels, val_mae_curve={3: float("nan")})

    inner_train = X_train[:, :-holdout_h]
    inner_val = X_train[:, -holdout_h:]
    inner_dist = dtw_distance_matrix(zscore_rows(inner_train))

    curve: Dict[int, float] = {}
    best_score = float("inf")
    best_k = max(k_grid[0], 2)
    best_labels: Optional[np.ndarray] = None

    for k in k_grid:
        if k * min_size > X_train.shape[0]:
            continue
        labels = cluster_with_min_size(inner_dist, k, min_size=min_size, seed=seed)
        score = score_fn(inner_train, inner_val, labels)
        curve[int(labels.max() + 1)] = float(score)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_k = int(labels.max() + 1)
            best_labels = labels

    if best_labels is None:
        # Defensive fall-back if every candidate produced NaN.
        best_labels = cluster_with_min_size(distance_matrix, 3, min_size=min_size, seed=seed)
        best_k = int(best_labels.max() + 1)

    # Re-cluster on full training data with the chosen k for final use.
    final_labels = cluster_with_min_size(distance_matrix, best_k, min_size=min_size, seed=seed)
    return ClusterSelectionResult(k=int(final_labels.max() + 1), labels=final_labels, val_mae_curve=curve)
