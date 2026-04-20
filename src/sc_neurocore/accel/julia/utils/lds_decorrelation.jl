# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/lds_decorrelation

module LdsDecorrelationAccel

using Statistics, LinearAlgebra

function generate_decorrelated_bitstreams(probabilities, length, method, seed)
    probabilities: np.ndarray,
    length: int = 1024,
    method: str = "sobol",
    seed: int | nothing = nothing,
    ) -> np.ndarray
    probs = np.asarray(probabilities, dtype=np.float64)
    flat_probs = probs.flatten()
    n_dims = length(flat_probs)
    if n_dims == 0
        return zeros((*probs.shape, length), dtype=np.uint8)
    if method == "sobol"
        sampler = qmc.Sobol(d=n_dims, seed=seed)
        samples = sampler.random(n=length)  # (length, n_dims)
    elseif method == "halton"
        sampler = qmc.Halton(d=n_dims, seed=seed)
        samples = sampler.random(n=length)  # (length, n_dims)
    else
        raise ValueError(f"Unknown method: {method}. Use 'sobol' || 'halton'.")
    # Threshold each dimension against its probability
    bits = zeros((n_dims, length), dtype=np.uint8)
    for d in 1:n_dims
        p = float(clamp(flat_probs[d], 0.0, 1.0))
        bits[d] = (samples[:, d] < p).astype(np.uint8)
    return bits.reshape(*probs.shape, length)
end

function star_discrepancy_estimate(samples, n_test)
    samples: np.ndarray,
    n_test: int = 10000,
    ) -> float
    n, d = samples.shape
    rng = np.random.RandomState(42)
    test_points = rng.uniform(0, 1, (n_test, d))
    max_disc = 0.0
    for pt in test_points
        # Fraction of samples in [0, pt] hypercube
        inside = np.all(samples <= pt, axis=1)
        empirical = mean(inside)
        # Volume of [0, pt] hypercube
        volume = np.prod(pt)
        disc = abs(empirical - volume)
        if disc > max_disc
            max_disc = disc
    return float(max_disc)
end

end # module LdsDecorrelationAccel
