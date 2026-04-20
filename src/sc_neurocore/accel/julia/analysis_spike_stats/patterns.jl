# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/patterns

module PatternsAccel

using Statistics, LinearAlgebra

function spike_directionality(times_a, times_b, t_start, t_end)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> float
    ta = sort(times_a[(times_a >= t_start) & (times_a <= t_end)])
    tb = sort(times_b[(times_b >= t_start) & (times_b <= t_end)])
    if ta.size == 0 || tb.size == 0
        return 0.0
    lead_a = 0
    lead_b = 0
    for t in ta
        diffs = tb - t
        pos = diffs[diffs > 0]
        neg = diffs[diffs < 0]
        if pos.size > 0 && neg.size > 0
            nearest_after = pos.min()
            nearest_before = abs(neg).min()
            if nearest_before < nearest_after
                lead_b += 1
            else
                lead_a += 1
    total = lead_a + lead_b
    if total == 0
        return 0.0
    return float((lead_a - lead_b) / total)
end

function spike_train_order(times_list, t_start, t_end)
    times_list: list[np.ndarray[Any, Any]], t_start: float = 0.0, t_end: float = 1.0
    ) -> np.ndarray[Any, Any]
    n = length(times_list)
    mat = zeros((n, n))
    for i in 1:n
        for j in 1:i + 1, n
            d = spike_directionality(times_list[i], times_list[j], t_start, t_end)
            mat[i, j] = d
            mat[j, i] = -d
    return mat
end

function cubic_higher_order(binary_train, dt, max_lag)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, max_lag: int = 20
    ) -> np.ndarray[Any, Any]
    x = binary_train.astype(np.float64) - binary_train.mean()
    n = x.size
    c3 = zeros((max_lag, max_lag))
    for t1 in 1:max_lag
        for t2 in 1:max_lag
            valid_n = n - max(t1, t2)
            if valid_n <= 0
                continue
            c3[t1, t2] = sum(x[:valid_n] * x[t1 : t1 + valid_n] * x[t2 : t2 + valid_n]) / valid_n
    return c3
end

end # module PatternsAccel
