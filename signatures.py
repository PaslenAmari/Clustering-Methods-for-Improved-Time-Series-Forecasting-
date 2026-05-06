"""
Signature representations used by the clustering ablation.

Each function maps a panel ``X`` of shape ``(n_series, n_timesteps)``
to a feature matrix ``S`` of shape ``(n_series, k)``.  Pairwise
distance between rows of ``S`` is computed by an algorithm-specific
helper in ``ablation.py``; this module is intentionally distance-free
so the same signature can be paired with different distances.
"""

from __future__ import annotations

import numpy as np


def signature_raw(X: np.ndarray) -> np.ndarray:
    """z-scored full series; intended for DTW or Euclidean."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / std


def signature_weekly_profile(X: np.ndarray) -> np.ndarray:
    """Average value per day-of-week, length-7 vector per series.

    Uses the assumption that the panel is already aligned to a daily
    grid (it is, by ``data_io``).  Profiles are z-scored within each
    series so clustering responds to weekly *shape* rather than level.
    """
    n_series, n_t = X.shape
    weeks = n_t // 7
    if weeks == 0:
        return signature_raw(X)
    truncated = X[:, : weeks * 7].reshape(n_series, weeks, 7)
    profile = truncated.mean(axis=1)
    mean = profile.mean(axis=1, keepdims=True)
    std = profile.std(axis=1, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (profile - mean) / std


def signature_spectral_envelope(X: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Magnitudes of the ``top_k`` FFT components after detrending."""
    n_series, n_t = X.shape
    detrended = X - X.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(detrended, axis=1))
    if spectrum.shape[1] >= top_k + 1:
        spectrum = spectrum[:, 1 : top_k + 1]  # drop DC, keep top_k
    norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return spectrum / norm


def signature_summary_stats(X: np.ndarray) -> np.ndarray:
    """Compact statistical signature: mean, std, ACF at lags 1 and 7,
    skewness, kurtosis, normalized entropy of binned values.

    Used as a lightweight stand-in for ``catch22``-style features when
    that library is unavailable.
    """
    from scipy.stats import skew, kurtosis

    n_series, n_t = X.shape
    feats = np.zeros((n_series, 7), dtype=float)
    for i in range(n_series):
        x = X[i]
        x_mean = x.mean()
        x_std = x.std() + 1e-12
        x_z = (x - x_mean) / x_std
        feats[i, 0] = x_mean
        feats[i, 1] = x_std
        feats[i, 2] = float(np.corrcoef(x_z[:-1], x_z[1:])[0, 1]) if n_t > 1 else 0.0
        feats[i, 3] = float(np.corrcoef(x_z[:-7], x_z[7:])[0, 1]) if n_t > 7 else 0.0
        feats[i, 4] = float(skew(x))
        feats[i, 5] = float(kurtosis(x))
        hist, _ = np.histogram(x_z, bins=10, density=True)
        hist = hist[hist > 0]
        feats[i, 6] = float(-np.sum(hist * np.log(hist + 1e-12)))
    # Normalize columns so distances are not dominated by raw scale.
    feats -= feats.mean(axis=0, keepdims=True)
    col_std = feats.std(axis=0, keepdims=True)
    col_std = np.where(col_std > 1e-12, col_std, 1.0)
    return feats / col_std


SIGNATURE_REGISTRY = {
    "raw": signature_raw,
    "weekly_profile": signature_weekly_profile,
    "spectral_envelope": signature_spectral_envelope,
    "summary_stats": signature_summary_stats,
}
