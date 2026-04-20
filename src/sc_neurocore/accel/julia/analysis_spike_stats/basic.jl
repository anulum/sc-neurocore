# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/basic

module BasicAccel

using Statistics, LinearAlgebra

function spike_times(binary_train, dt)
    return findall(binary_train > 0)[0] * dt
end

function isi(binary_train, dt)
    times = spike_times(binary_train, dt)
    if times.size < 2
        return collect([], dtype=np.float64)
    return diff(times)
end

function firing_rate(binary_train, dt)
    duration = binary_train.size * dt
    if duration <= 0
        return 0.0
    return float(sum(binary_train) / duration)
end

function spike_count(binary_train)
    return int(sum(binary_train))
end

function bin_spike_train(binary_train, bin_size)
    n = binary_train.size
    n_bins = n // bin_size
    if n_bins == 0
        return collect([int(binary_train.sum())])
    trimmed = binary_train[: n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size).sum(axis=1)
end

end # module BasicAccel
