# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bootstrap and permutation significance testing."""

from __future__ import annotations

import numpy as np


def significance_bootstrap(
    statistic_func,
    train_a: np.ndarray,
    train_b: np.ndarray,
    n_surrogates: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap significance test for a pairwise statistic.

    Returns (observed_value, p_value).
    statistic_func(a, b) -> float.
    """
    observed = statistic_func(train_a, train_b)
    rng = np.random.default_rng(seed)
    combined = np.concatenate([train_a, train_b])
    n_a = train_a.size
    count_extreme = 0
    for _ in range(n_surrogates):
        perm = rng.permutation(combined.size)
        surr_a = combined[perm[:n_a]]
        surr_b = combined[perm[n_a:]]
        surr_val = statistic_func(surr_a, surr_b)
        if abs(surr_val) >= abs(observed):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_surrogates + 1)
    return float(observed), float(p_value)
