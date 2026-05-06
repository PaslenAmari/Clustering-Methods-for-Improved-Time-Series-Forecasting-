"""
Integration smoke test: NN5 download + signatures + DTW + clustering
with the min_size=3 contract.

Run this once before the full pipeline to verify the environment is
healthy.  Requires only the lightweight dependency stack
(numpy + pandas + requests + scipy + scikit-learn + dtaidistance);
ETNA / CatBoost are not exercised here.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    from data_io import load_nn5
    from signatures import SIGNATURE_REGISTRY
    from clustering import cluster_with_min_size, dtw_distance_matrix, zscore_rows

    print("[1/4] Downloading and parsing NN5 ...")
    X, names, start_date = load_nn5(local_dir="../data/nn5")
    assert X is not None, "NN5 loader returned None"
    assert X.shape == (111, 791), f"unexpected NN5 shape {X.shape}"
    assert int(np.isfinite(X).sum()) == X.size, "NN5 contains NaNs after impute"
    print(f"      shape={X.shape}, start={start_date}, names={len(names)}")

    print("[2/4] Computing all signatures ...")
    for name, fn in SIGNATURE_REGISTRY.items():
        S = fn(X)
        assert np.isfinite(S).all(), f"signature {name} produced NaNs"
        print(f"      {name}: {S.shape}")

    print("[3/4] DTW distance matrix on full panel ...")
    t = time.time()
    D = dtw_distance_matrix(zscore_rows(X))
    elapsed = time.time() - t
    assert D.shape == (111, 111)
    assert (D == D.T).all(), "DTW matrix not symmetric"
    assert (D.diagonal() == 0).all(), "DTW diagonal not zero"
    print(f"      shape={D.shape}, mean={D.mean():.3f}, time={elapsed:.1f}s")

    print("[4/4] Clustering at k in {3,4,5,6,8} with min_size=3 ...")
    for k in (3, 4, 5, 6, 8):
        labels = cluster_with_min_size(D, n_clusters=k, min_size=3, seed=42)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        assert min(sizes) >= 3, f"min_size=3 violated at k={k}: sizes={sizes}"
        assert len(sizes) <= k, f"got {len(sizes)} clusters when {k} requested"
        print(f"      k_req={k} -> k_actual={len(sizes)}, sizes={sizes}")

    print("\n[OK] Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
