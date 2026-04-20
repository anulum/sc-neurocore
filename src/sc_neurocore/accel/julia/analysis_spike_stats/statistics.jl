# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/statistics

module StatisticsAccel

using Statistics, LinearAlgebra

function significance_bootstrap(statistic_func, train_a, train_b, n_surrogates, seed)
    statistic_func: Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float],
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    n_surrogates: int = 200,
    seed: int = 42,
    ) -> tuple[float, float]
    observed = statistic_func(train_a, train_b)
    rng = np.random.default_rng(seed)
    combined = vcat([train_a, train_b])
    n_a = train_a.size
    count_extreme = 0
    for _ in 1:n_surrogates
        perm = rng.permutation(combined.size)
        surr_a = combined[perm[:n_a]]
        surr_b = combined[perm[n_a:]]
        surr_val = statistic_func(surr_a, surr_b)
        if abs(surr_val) >= abs(observed)
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_surrogates + 1)
    return float(observed), float(p_value)
end

end # module StatisticsAccel
