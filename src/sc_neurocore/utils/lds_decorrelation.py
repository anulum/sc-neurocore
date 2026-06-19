# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Low-discrepancy sequence decorrelation for SC bitstreams

"""Multi-dimensional Sobol/Halton decorrelation for per-synapse independent streams.

In SC, correlation between bitstreams causes computation errors. Using
independent random streams per synapse is expensive (N*M RNG calls for
an N×M weight matrix). Low-discrepancy sequences (LDS) provide
decorrelated streams with better convergence:

- Sobol: O(log(N)^d / N) discrepancy in d dimensions
- Halton: O(log(N)^d / N) using Van der Corput bases

Each synapse (i, j) gets dimension d = i*M + j of the LDS, ensuring
decorrelation across the entire weight matrix with a single sequence.

    from sc_neurocore.utils.lds_decorrelation import generate_decorrelated_bitstreams

    # 4 inputs × 3 neurons = 12 decorrelated streams
    streams = generate_decorrelated_bitstreams(
        probabilities=weight_matrix,  # shape (3, 4)
        length=1024,
        method="sobol",
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats.qmc as qmc


def generate_decorrelated_bitstreams(
    probabilities: np.ndarray[Any, Any],
    length: int = 1024,
    method: str = "sobol",
    seed: int | None = None,
) -> np.ndarray[Any, Any]:
    """Generate decorrelated bitstreams for a probability matrix.

    Each element of the probability matrix gets its own LDS dimension,
    ensuring zero correlation between any pair of bitstreams.

    Parameters
    ----------
    probabilities : np.ndarray[Any, Any]
        Probability matrix, any shape. Values in [0, 1].
    length : int
        Bitstream length per element.
    method : str
        "sobol" or "halton".
    seed : int or None
        Random seed for scrambling.

    Returns
    -------
    np.ndarray[Any, Any]
        Shape (*probabilities.shape, length), dtype uint8.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    flat_probs = probs.flatten()
    n_dims = len(flat_probs)

    if n_dims == 0:
        return np.zeros((*probs.shape, length), dtype=np.uint8)

    if method == "sobol":
        sampler = qmc.Sobol(d=n_dims, seed=seed)
        samples = sampler.random(n=length)  # (length, n_dims)
    elif method == "halton":
        sampler = qmc.Halton(d=n_dims, seed=seed)
        samples = sampler.random(n=length)  # (length, n_dims)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sobol' or 'halton'.")

    # Threshold each dimension against its probability
    bits = np.zeros((n_dims, length), dtype=np.uint8)
    for d in range(n_dims):
        p = float(np.clip(flat_probs[d], 0.0, 1.0))
        bits[d] = (samples[:, d] < p).astype(np.uint8)

    return bits.reshape(*probs.shape, length)


def star_discrepancy_estimate(
    samples: np.ndarray[Any, Any],
    n_test: int = 10000,
) -> float:
    """Estimate star discrepancy of a sample set (quality metric for LDS).

    Lower discrepancy → more uniform coverage → better SC precision.

    Parameters
    ----------
    samples : np.ndarray[Any, Any]
        Shape (n_samples, d), values in [0, 1].
    n_test : int
        Number of random test points.

    Returns
    -------
    float
        Estimated star discrepancy.
    """
    n, d = samples.shape
    rng = np.random.RandomState(42)
    test_points = rng.uniform(0, 1, (n_test, d))

    max_disc = 0.0
    for pt in test_points:
        # Fraction of samples in [0, pt] hypercube
        inside = np.all(samples <= pt, axis=1)
        empirical = np.mean(inside)
        # Volume of [0, pt] hypercube
        volume = np.prod(pt)
        disc = abs(empirical - volume)
        if disc > max_disc:
            max_disc = disc

    return float(max_disc)
