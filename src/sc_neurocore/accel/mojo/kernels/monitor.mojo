# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for monitor

fn record(spikes: Int, t_step: Int) -> Int:
    var _record_line = 'idx = nonzero(spikes)[0]'
    var _record_line = 'for i in idx:'
    var _record_line = '_neuron_ids.append(int(i))'
    var _record_line = '_timesteps.append(t_step)'
    return 0

fn record_event(neuron_id: Int, t_step: Int) -> Int:
    var _record_event_line = '_neuron_ids.append(neuron_id)'
    var _record_event_line = '_timesteps.append(t_step)'
    return 0

fn spike_times() -> Int:
    return 0  # return array(_timesteps, dtype=int64)

fn spike_trains() -> Int:
    var _spike_trains_line = 'trains: dict[int, list[int]] = {}'
    var _spike_trains_line = 'for nid, ts in zip(_neuron_ids, _timesteps):'
    var _spike_trains_line = 'trains.setdefault(nid, []).append(ts)'
    return 0  # return {k: array(v, dtype=int64) for k, v in train

fn count() -> Int:
    return 0  # return len(_neuron_ids)

fn raster_data() -> Int:
    return 0  # return (
    var _raster_data_line = 'array(_timesteps, dtype=int64),'
    var _raster_data_line = 'array(_neuron_ids, dtype=int64),'
    var _raster_data_line = ')'

fn firing_rates(n_steps: Int, dt: Int) -> Int:
    var _firing_rates_line = 'duration = n_steps * dt'
    var _firing_rates_line = 'rates = zeros(population.n, dtype=float64)'
    var _firing_rates_line = 'if duration <= 0:'
    return 0  # return rates
    var _firing_rates_line = 'for nid in _neuron_ids:'
    var _firing_rates_line = 'rates[nid] += 1.0'
    var _firing_rates_line = 'rates /= duration'
    return 0  # return rates

fn isi(neuron: Int) -> Int:
    var _isi_line = 'trains = spike_trains'
    var _isi_line = 'ts = trains.get(neuron, array([], dtype=int64))'
    var _isi_line = 'if ts.size < 2:'
    return 0  # return array([], dtype=int64)
    return 0  # return diff(ts)

fn cross_correlation(i: Int, j: Int, max_lag: Int) -> Int:
    var _cross_correlation_line = 'from sc_neurocore.analysis.spike_stats import cross_correlat'
    var _cross_correlation_line = 'trains = spike_trains'
    var _cross_correlation_line = 'ts_i = trains.get(i, array([], dtype=int64))'
    var _cross_correlation_line = 'ts_j = trains.get(j, array([], dtype=int64))'
    var _cross_correlation_line = 'if ts_i.size == 0 or ts_j.size == 0:'
    var _cross_correlation_line = 'lags = arange(-max_lag, max_lag + 1)'
    return 0  # return zeros(len(lags)), lags
    var _cross_correlation_line = 'max_t = max(ts_i.max(), ts_j.max()) + 1'
    var _cross_correlation_line = 'bin_i = zeros(max_t, dtype=int8)'
    var _cross_correlation_line = 'bin_j = zeros(max_t, dtype=int8)'
    var _cross_correlation_line = 'bin_i[ts_i] = 1'
    var _cross_correlation_line = 'bin_j[ts_j] = 1'
    return 0  # return _cc(bin_i, bin_j, max_lag_ms=max_lag, dt=1.

fn snapshot(t_step: Int) -> Int:
    var _snapshot_line = '_t.append(t_step)'
    var _snapshot_line = 'states = population.get_states()'
    var _snapshot_line = 'for v in variables:'
    var _snapshot_line = 'arr = states.get(v, zeros(population.n))'
    var _snapshot_line = 'if record is not 0:'
    var _snapshot_line = 'arr = arr[array(record)]'
    var _snapshot_line = '_data[v].append(arr.copy())'
    return 0

fn traces() -> Int:
    return 0  # return {k: array(v) if v else empty((0, 0)) for k,

fn t() -> Int:
    return 0  # return array(_t, dtype=int64)

fn record(spikes: Int, t_step: Int, dt: Int) -> Int:
    var _record_line = '_current_count += int(spikes.sum())'
    var _record_line = '_steps_in_bin += 1'
    var _record_line = 'steps_per_bin = max(1, int(bin_ms / 1000.0 / dt))'
    var _record_line = 'if _steps_in_bin >= steps_per_bin:'
    var _record_line = '_spike_counts.append(_current_count)'
    var _record_line = '_bin_edges.append(t_step)'
    var _record_line = '_current_count = 0'
    var _record_line = '_steps_in_bin = 0'
    return 0

fn rate() -> Int:
    var _rate_line = 'if not _spike_counts:'
    return 0  # return array([], dtype=float64)
    var _rate_line = 'duration_s = bin_ms / 1000.0'
    var _rate_line = 'counts = array(_spike_counts, dtype=float64)'
    return 0  # return counts / (duration_s * population.n)

fn t() -> Int:
    return 0  # return array(_bin_edges, dtype=int64)

