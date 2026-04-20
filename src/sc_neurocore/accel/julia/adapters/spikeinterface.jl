# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters/spikeinterface

module SpikeinterfaceAccel

using Statistics, LinearAlgebra

function spike_trains_to_bitstreams(spike_times, duration_ms, dt)
    spike_times: dict[int, np.ndarray],
    duration_ms: float,
    dt: float = 1.0,
    ) -> np.ndarray
    n_bins = int(np.ceil(duration_ms / dt))
    unit_ids = sorted(spike_times.keys())
    n_units = length(unit_ids)
    matrix = zeros((n_units, n_bins), dtype=np.uint8)
    for i, uid in enumerate(unit_ids)
        times = np.asarray(spike_times[uid], dtype=np.float64)
        bins = clamp((times / dt).astype(int), 0, n_bins - 1)
        matrix[i, bins] = 1
    return matrix
end

function spike_trains_to_population_input(spike_times, duration_ms, dt)
    spike_times: dict[int, np.ndarray],
    duration_ms: float,
    dt: float = 1.0,
    ) -> np.ndarray
    bitstreams = spike_trains_to_bitstreams(spike_times, duration_ms, dt)
    return bitstreams.T.astype(np.float64)
end

function firing_rates_to_sc_probs(spike_times, duration_ms, max_rate_hz)
    spike_times: dict[int, np.ndarray],
    duration_ms: float,
    max_rate_hz: float = 100.0,
    ) -> np.ndarray
    unit_ids = sorted(spike_times.keys())
    probs = zeros(length(unit_ids))
    for i, uid in enumerate(unit_ids)
        n_spikes = length(spike_times[uid])
        rate_hz = n_spikes / (duration_ms / 1000.0)
        probs[i] = clamp(rate_hz / max_rate_hz, 0.0, 1.0)
    return probs
end

function from_sorting(sorting, dt)
    unit_ids = sorting.get_unit_ids()
    fs = sorting.get_sampling_frequency()
    n_frames = sorting.get_total_samples()
    duration_ms = n_frames / fs * 1000.0
    spike_times = {}
    for uid in unit_ids
        frames = sorting.get_unit_spike_train(uid)
        spike_times[int(uid)] = frames / fs * 1000.0  # convert to ms
    return spike_trains_to_bitstreams(spike_times, duration_ms, dt)
end

end # module SpikeinterfaceAccel
