"""Minimal sanity tests for eval.py (numpy/scipy only).

Verifies the per-series aggregation contract and the corrected
Wilcoxon-statistic upper bound.
"""

import sys

import numpy as np

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


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.5, 2.5, 2.5])
    assert _approx(mae(y, yhat), 0.5), mae(y, yhat)
    assert smape(y, yhat) > 0
    history = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 2.0])
    val = mase(np.array([1.5]), np.array([1.0]), history, seasonality=3)
    assert val > 0, val


def test_wilcoxon_upper_bound():
    """Reproduce the bug we are fixing: with N paired series the Wilcoxon
    W must be <= N(N+1)/2.  For N=42 that is 903, never 11633."""
    rng = np.random.default_rng(0)
    n = 42
    baseline = rng.uniform(1.0, 2.0, size=n)
    method = baseline - rng.uniform(0.0, 0.3, size=n)
    res = paired_test(method, baseline, label="sanity")
    upper = n * (n + 1) / 2
    assert 0 <= res.wilcoxon_stat <= upper, (res.wilcoxon_stat, upper)
    assert 0.0 <= res.win_rate <= 1.0
    assert -1.0 <= res.cliffs_delta <= 1.0
    assert res.bootstrap_ci_low <= res.median_improvement <= res.bootstrap_ci_high


def test_holm_monotonic():
    raw = [0.001, 0.04, 0.03, 0.5, 0.0001]
    adjusted = holm_correction(raw)
    assert all(0 <= p <= 1 for p in adjusted)
    # Adjusted p-values must dominate raw ones
    assert all(adj >= r - 1e-12 for adj, r in zip(adjusted, raw))


def test_aggregate_per_series_means_across_windows():
    windows = [
        {"a": 1.0, "b": 2.0},
        {"a": 3.0, "b": 4.0},
    ]
    agg = aggregate_per_series(windows)
    assert _approx(agg["a"], 2.0)
    assert _approx(agg["b"], 3.0)


def test_format_pvalue():
    assert format_pvalue(1e-7) == "<1e-4"
    assert format_pvalue(0.5).startswith("0.5")
    assert format_pvalue(float("nan")) == "n/a"


def test_bootstrap_ci_includes_median():
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.1, 0.05, size=200)
    lo, hi = bootstrap_median_ci(deltas, n_boot=2000)
    assert lo < float(np.median(deltas)) < hi


def main() -> int:
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [OK] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  [FAIL] {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
