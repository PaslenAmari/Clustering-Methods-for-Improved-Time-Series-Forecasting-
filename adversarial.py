"""
Adversarial random-cluster ablation (paper section 6.x).

For each backtest window of NN5 at horizons in {7, 14, 30}, the
DTW-derived partition and per-cluster model mapping are read from
``results/seed_NN5.json``.  The same `forecast_clustered_local`
pipeline is then re-run with `R = 10` random partitions whose cluster
sizes match the DTW partition exactly.  Per-series MAE is collected
for both the DTW and random arms; paired statistics are reported per
horizon.

Outputs:
    results/adversarial_NN5.csv    per (horizon, seed) summary
    results/adversarial_paired.csv per horizon DTW vs random aggregate
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data_io import load_dataset
from eval import format_pvalue, paired_test
from forecast import forecast_clustered_local
from run_damdid import _evaluate_window_forecasts


RESULTS_DIR = "results"
SEED_PATH = os.path.join(RESULTS_DIR, "seed_NN5.json")
R = 10


def _series_id(idx: int) -> str:
    return f"region_{idx}"


def random_partition_with_sizes(n: int, sizes: List[int], rng: np.random.Generator) -> np.ndarray:
    """Shuffle 0..n-1 and chunk into the requested cluster sizes."""
    assert sum(sizes) == n, f"sizes sum {sum(sizes)} != n {n}"
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=int)
    cursor = 0
    for cid, sz in enumerate(sizes):
        labels[perm[cursor : cursor + sz]] = cid
        cursor += sz
    return labels


def aggregate_per_series(window_records: List[Dict[str, float]]) -> Dict[str, float]:
    accumulator: Dict[str, List[float]] = {}
    for window in window_records:
        for sid, val in window.items():
            if not np.isfinite(val):
                continue
            accumulator.setdefault(sid, []).append(val)
    return {sid: float(np.mean(vals)) for sid, vals in accumulator.items() if vals}


def main() -> int:
    if not os.path.exists(SEED_PATH):
        print(f"  [!] {SEED_PATH} not found; run seed_runs.py first")
        return 1
    with open(SEED_PATH, "r", encoding="utf-8") as fh:
        seed = json.load(fh)
    print(f"Loaded seed: {seed['dataset']} shape={seed['shape']} horizons={list(seed['horizons'])}")

    X, _, start_date = load_dataset(seed["dataset"])
    assert X is not None and list(X.shape) == seed["shape"], "dataset shape changed since seed"

    summary_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []

    for horizon_str, h_block in seed["horizons"].items():
        horizon = int(horizon_str)
        windows = h_block["windows"]
        print(f"\n=== h={horizon} ===")

        # DTW per-series MAE (already in seed)
        dtw_per_window = [w["cl_per_series_mae"] for w in windows]
        dtw_per_series = aggregate_per_series(dtw_per_window)
        dtw_mean = float(np.mean(list(dtw_per_series.values())))
        print(f"  DTW CL-Occam mean MAE = {dtw_mean:.3f}")

        # R random seeds
        rand_per_seed_mae: Dict[int, Dict[str, float]] = {}
        for seed_idx in range(R):
            rng = np.random.default_rng(1000 + seed_idx)
            window_records: List[Dict[str, float]] = []
            for w in windows:
                test_start = int(w["test_start"])
                test_end = int(w["test_end"])
                X_train = X[:, :test_start]
                sizes = w["cluster_sizes"]
                labels = random_partition_with_sizes(X.shape[0], sizes, rng)
                # Re-map model assignment: keep the *distribution* of model
                # choices but reassign by random cluster id.  Order of mappings
                # in seed is arbitrary; we shuffle them too so a "Ridge" cell
                # need not align with the same DTW cluster id.
                src_models = list(w["model_mapping"].values())
                rng.shuffle(src_models)
                rand_mapping = {cid: src_models[cid] if cid < len(src_models) else "ridge" for cid in range(len(sizes))}

                forecasts = forecast_clustered_local(X_train, labels, rand_mapping, start_date, horizon)
                errs = _evaluate_window_forecasts(forecasts, X, test_start, test_end, X_train)
                window_records.append({seg: m["mae"] for seg, m in errs.items()})
            seed_per_series = aggregate_per_series(window_records)
            mean_mae = float(np.mean(list(seed_per_series.values()))) if seed_per_series else float("nan")
            print(f"  random seed {seed_idx}: mean MAE = {mean_mae:.3f}")
            rand_per_seed_mae[seed_idx] = seed_per_series
            summary_rows.append({"horizon": horizon, "seed": seed_idx, "mean_mae": mean_mae})

        # Paired test: DTW vs the *average* random per series (one paired sample per series).
        shared = sorted(set(dtw_per_series).intersection(*[set(d) for d in rand_per_seed_mae.values()]))
        if not shared:
            print("  [!] no shared series; skipping paired test")
            continue
        rand_avg_per_series = {
            sid: float(np.mean([rand_per_seed_mae[r][sid] for r in rand_per_seed_mae])) for sid in shared
        }
        dtw_arr = np.array([dtw_per_series[s] for s in shared])
        rand_arr = np.array([rand_avg_per_series[s] for s in shared])
        result = paired_test(dtw_arr, rand_arr, label=f"DTW vs random h={horizon}")

        rand_means = [
            float(np.mean(list(rand_per_seed_mae[r].values()))) for r in rand_per_seed_mae
        ]
        rand_mean = float(np.mean(rand_means))
        rand_std = float(np.std(rand_means))
        print(f"  random mean MAE across seeds = {rand_mean:.3f} +/- {rand_std:.3f}")
        print(f"  Wilcoxon p (DTW < random_avg) = {format_pvalue(result.wilcoxon_p)}, "
              f"Cliff's delta = {result.cliffs_delta:+.3f}")

        paired_rows.append(
            {
                "horizon": horizon,
                "n_series": result.n,
                "dtw_mean_mae": dtw_mean,
                "random_mean_mae": rand_mean,
                "random_std_mae": rand_std,
                "median_improvement": result.median_improvement,
                "ci_low": result.bootstrap_ci_low,
                "ci_high": result.bootstrap_ci_high,
                "win_rate": result.win_rate,
                "wilcoxon_stat": result.wilcoxon_stat,
                "wilcoxon_p": result.wilcoxon_p,
                "wilcoxon_p_str": format_pvalue(result.wilcoxon_p),
                "cliffs_delta": result.cliffs_delta,
            }
        )

    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "adversarial_NN5.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(RESULTS_DIR, "adversarial_paired.csv"), index=False)
    print("\nWrote results/adversarial_NN5.csv and results/adversarial_paired.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
