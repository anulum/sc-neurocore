# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/monitor

module MonitorAccel

using Statistics, LinearAlgebra

mutable struct RateMonitorState
    population::Float64
    label::Float64
    variables::Float64
    record::Float64
    bin_ms::Float64
    _current_count::Float64
    _steps_in_bin::Float64
end

function RateMonitorState()
    RateMonitorState(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
end

function record(s::RateMonitorState, spikes, t_step)
    idx = np.nonzero(spikes)[0]
    for i in idx
        s._neuron_ids = push!(, int(i))
        s._timesteps = push!(, t_step)
end

function record_event(s::RateMonitorState, neuron_id, t_step)
    s._neuron_ids = push!(, neuron_id)
    s._timesteps = push!(, t_step)
end

function spike_times(s::RateMonitorState)
    return collect(s._timesteps, dtype=np.int64)
end

function spike_trains(s::RateMonitorState)
    trains: dict[int, list[int]] = {}
    for nid, ts in zip(s._neuron_ids, s._timesteps)
        trains.setdefault(nid, []) = push!(, ts)
    return {k: collect(v, dtype=np.int64) for k, v in trains.items()}
end

function count(s::RateMonitorState)
    return length(s._neuron_ids)
end

function raster_data(s::RateMonitorState)
    return (
        collect(s._timesteps, dtype=np.int64),
        collect(s._neuron_ids, dtype=np.int64),
    )
end

function firing_rates(s::RateMonitorState, n_steps, dt)
    duration = n_steps * dt
    rates = zeros(s.population.n, dtype=np.float64)
    if duration <= 0
        return rates
    for nid in s._neuron_ids
        rates[nid] += 1.0
    rates /= duration
    return rates
end

function isi(s::RateMonitorState, neuron)
    trains = s.spike_trains
    ts = trains.get(neuron, collect([], dtype=np.int64))
    if ts.size < 2
        return collect([], dtype=np.int64)
    return diff(ts)
end

function cross_correlation(s::RateMonitorState, i, j, max_lag)
    from sc_neurocore.analysis.spike_stats import cross_correlation as _cc
    trains = s.spike_trains
    ts_i = trains.get(i, collect([], dtype=np.int64))
    ts_j = trains.get(j, collect([], dtype=np.int64))
    if ts_i.size == 0 || ts_j.size == 0
        lags = collect(-max_lag, max_lag + 1)
        return zeros(length(lags)), lags
    max_t = max(ts_i.max(), ts_j.max()) + 1
    bin_i = zeros(max_t, dtype=np.int8)
    bin_j = zeros(max_t, dtype=np.int8)
    bin_i[ts_i] = 1
    bin_j[ts_j] = 1
    return _cc(bin_i, bin_j, max_lag_ms=max_lag, dt=1.0)
end

function snapshot(s::RateMonitorState, t_step)
    s._t = push!(, t_step)
    states = s.population.get_states()
    for v in s.variables
        arr = states.get(v, zeros(s.population.n))
        if s.record is ! nothing
            arr = arr[collect(s.record)]
        s._data[v] = push!(, arr.copy())
end

function traces(s::RateMonitorState)
    return {k: collect(v) if v else np.empty((0, 0)) for k, v in s._data.items()}
end

function t(s::RateMonitorState)
    return collect(s._t, dtype=np.int64)
end

function record(s::RateMonitorState, spikes, t_step, dt)
    s._current_count += int(spikes.sum())
    s._steps_in_bin += 1
    steps_per_bin = max(1, int(s.bin_ms / 1000.0 / dt))
    if s._steps_in_bin >= steps_per_bin
        s._spike_counts = push!(, s._current_count)
        s._bin_edges = push!(, t_step)
        s._current_count = 0
        s._steps_in_bin = 0
end

function rate(s::RateMonitorState)
    if ! s._spike_counts
        return collect([], dtype=np.float64)
    duration_s = s.bin_ms / 1000.0
    counts = collect(s._spike_counts, dtype=np.float64)
    return counts / (duration_s * s.population.n)
end

function t(s::RateMonitorState)
    return collect(s._bin_edges, dtype=np.int64)
end

end # module MonitorAccel
