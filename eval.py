"""
Forecast quality metrics and paired statistical tests.

Two design rules followed throughout:

* Per-series aggregation comes first.  Forecast errors collected across
  multiple backtest windows for the same series are averaged into a
  single per-series score before any paired test runs.  Treating each
  (series, window) cell as independent inflates apparent significance,
  which we explicitly avoid here.
* Effect-size reporting alongside p-values.  Each comparison emits a
  median improvement with a bootstrap interval and Cliff's delta, so
  practical significance is never confused with statistical noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Pointwise metrics
# ---------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_history: np.ndarray,
    seasonality: int = 7,
) -> float:
    """Mean Absolute Scaled Error with a seasonal-naive denominator.

    The denominator is computed once on the training history of the
    same series, so it is independent of the test window length and
    invariant to forecast horizon.
    """
    if len(train_history) <= seasonality:
        return float("nan")
    seasonal_diff = np.abs(train_history[seasonality:] - train_history[:-seasonality])
    denom = float(np.mean(seasonal_diff))
    if denom <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


@dataclass
class PairedTestResult:
    n: int
    median_baseline: float
    median_method: float
    median_improvement: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    win_rate: float
    wilcoxon_stat: float
    wilcoxon_p: float
    cliffs_delta: float
    holm_p: Optional[float] = None
    label: str = ""

    def as_row(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "n": self.n,
            "median_baseline": self.median_baseline,
            "median_method": self.median_method,
            "median_improvement": self.median_improvement,
            "ci_low": self.bootstrap_ci_low,
            "ci_high": self.bootstrap_ci_high,
            "win_rate": self.win_rate,
            "wilcoxon_stat": self.wilcoxon_stat,
            "wilcoxon_p": self.wilcoxon_p,
            "cliffs_delta": self.cliffs_delta,
            "holm_p": self.holm_p,
        }


def cliffs_delta(method: np.ndarray, baseline: np.ndarray) -> float:
    """Cliff's delta: P(method < baseline) - P(method > baseline).

    Positive values indicate the method has lower errors than the
    baseline, since both arrays are interpreted as error magnitudes.
    """
    method = np.asarray(method)
    baseline = np.asarray(baseline)
    n = len(method) * len(baseline)
    if n == 0:
        return float("nan")
    diff = method[:, None] - baseline[None, :]
    less = float(np.sum(diff < 0))
    greater = float(np.sum(diff > 0))
    return (less - greater) / n


def bootstrap_median_ci(
    deltas: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    boots = np.median(deltas[idx], axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def paired_test(
    method_per_series: Sequence[float],
    baseline_per_series: Sequence[float],
    label: str = "",
    n_boot: int = 10000,
) -> PairedTestResult:
    """One-sided Wilcoxon (method < baseline) on per-series errors.

    Inputs must already be aggregated to one value per series.  This
    aggregation contract is the central correction over the original
    pipeline, which compared (series, window) pairs and so violated the
    independence assumption of the test.
    """
    method = np.asarray(method_per_series, dtype=float)
    baseline = np.asarray(baseline_per_series, dtype=float)
    if method.shape != baseline.shape:
        raise ValueError("method and baseline arrays must have identical length")

    mask = np.isfinite(method) & np.isfinite(baseline)
    method = method[mask]
    baseline = baseline[mask]
    if len(method) < 2:
        return PairedTestResult(
            n=len(method),
            median_baseline=float("nan"),
            median_method=float("nan"),
            median_improvement=float("nan"),
            bootstrap_ci_low=float("nan"),
            bootstrap_ci_high=float("nan"),
            win_rate=float("nan"),
            wilcoxon_stat=float("nan"),
            wilcoxon_p=float("nan"),
            cliffs_delta=float("nan"),
            label=label,
        )

    deltas_relative = (baseline - method) / np.where(np.abs(baseline) > 1e-12, baseline, np.nan)
    median_improvement = float(np.nanmedian(deltas_relative))
    ci_low, ci_high = bootstrap_median_ci(deltas_relative[np.isfinite(deltas_relative)], n_boot=n_boot)
    win_rate = float(np.mean(method < baseline))

    try:
        wilcoxon = scipy_stats.wilcoxon(baseline, method, alternative="greater", zero_method="wilcox")
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        # Wilcoxon raises when all paired differences are zero.
        wilcoxon_stat = float("nan")
        wilcoxon_p = 1.0

    return PairedTestResult(
        n=len(method),
        median_baseline=float(np.median(baseline)),
        median_method=float(np.median(method)),
        median_improvement=median_improvement,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        win_rate=win_rate,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        cliffs_delta=cliffs_delta(method, baseline),
        label=label,
    )


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction returning adjusted p-values.

    Order-preserving: the i-th input p-value maps to the i-th output.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate_per_series(window_errors: List[Dict[str, float]]) -> Dict[str, float]:
    """Collapse a list of per-window {series_id: error} maps to one
    error per series by mean.

    Series missing in any window are kept with the mean of the windows
    they do appear in; this matches the conservative aggregation used
    in the rest of the pipeline.
    """
    accumulator: Dict[str, List[float]] = {}
    for window in window_errors:
        for sid, val in window.items():
            if not np.isfinite(val):
                continue
            accumulator.setdefault(sid, []).append(val)
    return {sid: float(np.mean(vals)) for sid, vals in accumulator.items() if vals}


def format_pvalue(p: float) -> str:
    """Render p-values without ever printing 'p = 0.0000'."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"
