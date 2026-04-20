# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for spikeinterface

fn spike_trains_to_bitstreams(spike_times: Int, duration_ms: Int, dt: Int) -> Int:
    var _spike_trains_to_bitstreams_line = 'spike_times: dict[int, ndarray],'
    var _spike_trains_to_bitstreams_line = 'duration_ms: float,'
    var _spike_trains_to_bitstreams_line = 'dt: float = 1.0,'
    var _spike_trains_to_bitstreams_line = ') -> ndarray:'
    var _spike_trains_to_bitstreams_line = 'n_bins = int(ceil(duration_ms / dt))'
    var _spike_trains_to_bitstreams_line = 'unit_ids = sorted(spike_times.keys())'
    var _spike_trains_to_bitstreams_line = 'n_units = len(unit_ids)'
    var _spike_trains_to_bitstreams_line = 'matrix = zeros((n_units, n_bins), dtype=uint8)'
    var _spike_trains_to_bitstreams_line = 'for i, uid in enumerate(unit_ids):'
    var _spike_trains_to_bitstreams_line = 'times = asarray(spike_times[uid], dtype=float64)'
    var _spike_trains_to_bitstreams_line = 'bins = clip((times / dt).astype(int), 0, n_bins - 1)'
    var _spike_trains_to_bitstreams_line = 'matrix[i, bins] = 1'
    return 0  # return matrix

fn spike_trains_to_population_input(spike_times: Int, duration_ms: Int, dt: Int) -> Int:
    var _spike_trains_to_population_input_line = 'spike_times: dict[int, ndarray],'
    var _spike_trains_to_population_input_line = 'duration_ms: float,'
    var _spike_trains_to_population_input_line = 'dt: float = 1.0,'
    var _spike_trains_to_population_input_line = ') -> ndarray:'
    var _spike_trains_to_population_input_line = 'bitstreams = spike_trains_to_bitstreams(spike_times, duratio'
    return 0  # return bitstreams.T.astype(float64)

fn firing_rates_to_sc_probs(spike_times: Int, duration_ms: Int, max_rate_hz: Int) -> Int:
    var _firing_rates_to_sc_probs_line = 'spike_times: dict[int, ndarray],'
    var _firing_rates_to_sc_probs_line = 'duration_ms: float,'
    var _firing_rates_to_sc_probs_line = 'max_rate_hz: float = 100.0,'
    var _firing_rates_to_sc_probs_line = ') -> ndarray:'
    var _firing_rates_to_sc_probs_line = 'unit_ids = sorted(spike_times.keys())'
    var _firing_rates_to_sc_probs_line = 'probs = zeros(len(unit_ids))'
    var _firing_rates_to_sc_probs_line = 'for i, uid in enumerate(unit_ids):'
    var _firing_rates_to_sc_probs_line = 'n_spikes = len(spike_times[uid])'
    var _firing_rates_to_sc_probs_line = 'rate_hz = n_spikes / (duration_ms / 1000.0)'
    var _firing_rates_to_sc_probs_line = 'probs[i] = clip(rate_hz / max_rate_hz, 0.0, 1.0)'
    return 0  # return probs

fn from_sorting(sorting: Int, dt: Int) -> Int:
    var _from_sorting_line = 'unit_ids = sorting.get_unit_ids()'
    var _from_sorting_line = 'fs = sorting.get_sampling_frequency()'
    var _from_sorting_line = 'n_frames = sorting.get_total_samples()'
    var _from_sorting_line = 'duration_ms = n_frames / fs * 1000.0'
    var _from_sorting_line = 'spike_times = {}'
    var _from_sorting_line = 'for uid in unit_ids:'
    var _from_sorting_line = 'frames = sorting.get_unit_spike_train(uid)'
    var _from_sorting_line = 'spike_times[int(uid)] = frames / fs * 1000.0  # convert to m'
    return 0  # return spike_trains_to_bitstreams(spike_times, dur
