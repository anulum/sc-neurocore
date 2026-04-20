# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/phi_estimation

module PhiEstimationAccel

using Statistics, LinearAlgebra

function phi_star(data, tau)
    n, T = data.shape
    if 2 * tau >= T || n < 2
        return 0.0
    past = data[:, :-tau]
    future = data[:, tau:]
    # Joint mutual information I(past; future)
    mi_whole = _gaussian_mi(past, future)
    # Minimum information partition: try all bipartitions
    # For tractability, only try contiguous splits (first k vs rest)
    mi_parts_min = Inf
    for k in 1:1, n
        idx_a = list(range(k))
        idx_b = list(range(k, n))
        mi_a = _gaussian_mi(past[idx_a], future[idx_a])
        mi_b = _gaussian_mi(past[idx_b], future[idx_b])
        mi_parts_min = min(mi_parts_min, mi_a + mi_b)
    phi = max(0.0, mi_whole - mi_parts_min)
    return float(phi)
end

function phi_from_spike_trains(spikes, bin_size, tau)
    n_neurons, n_steps = spikes.shape
    n_bins = n_steps // bin_size
    if n_bins < 2 * tau + 2
        return 0.0
    # Bin spike trains into spike counts
    binned = zeros((n_neurons, n_bins))
    for b in 1:n_bins
        binned[:, b] = spikes[:, b * bin_size : (b + 1) * bin_size].sum(axis=1)
    return phi_star(binned, tau=tau)
end

end # module PhiEstimationAccel
