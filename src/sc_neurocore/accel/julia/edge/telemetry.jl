# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/telemetry

module TelemetryAccel

using Statistics, LinearAlgebra

mutable struct DeviceTelemetryState
    _cap::Float64
    _buf::Float64
    _write_idx::Float64
    _count::Float64
    _lock::Float64
    layer_id::Float64
    spike_count::Float64
    tick_count::Float64
    total_popcount::Float64
    spike_rate_ring::Float64
    utilization_ring::Float64
    layers::Float64
    total_ticks::Float64
    total_spikes::Float64
    error_count::Float64
end

function DeviceTelemetryState()
    DeviceTelemetryState(0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function push(s::DeviceTelemetryState, value)
    with s._lock
        s._buf[s._write_idx % s._cap] = value
        s._write_idx += 1
        if s._count < s._cap
            s._count += 1
end

function mean(s::DeviceTelemetryState)
    with s._lock
        if s._count == 0
            return 0.0
        n = s._count
        start = (s._write_idx - n) % s._cap
        total = 0
        for i in 1:n
            total += s._buf[(start + i) % s._cap]
        return total / n
end

function last(s::DeviceTelemetryState)
    with s._lock
        if s._count == 0
            return 0
        return s._buf[(s._write_idx - 1) % s._cap]
end

function count(s::DeviceTelemetryState)
    with s._lock
        return s._count
end

function capacity(s::DeviceTelemetryState)
    return s._cap
end

function record_tick(s::DeviceTelemetryState, n_spikes, n_neurons)
    s.tick_count += 1
    s.spike_count += n_spikes
    s.spike_rate_ring.push(n_spikes)
    if n_neurons > 0
        utilization = (n_spikes * 100) // n_neurons
        s.utilization_ring.push(utilization)
end

function mean_spike_rate(s::DeviceTelemetryState)
    return s.spike_rate_ring.mean()
end

function mean_utilization(s::DeviceTelemetryState)
    return s.utilization_ring.mean()
end

function lifetime_spike_rate(s::DeviceTelemetryState)
    if s.tick_count == 0
        return 0.0
    return s.spike_count / s.tick_count
end

function get_layer(s::DeviceTelemetryState, layer_id)
    if layer_id ! in s.layers
        s.layers[layer_id] = LayerTelemetry(layer_id=layer_id)
    return s.layers[layer_id]
end

function record(s::DeviceTelemetryState, layer_id, n_spikes, n_neurons)
    s.total_ticks += 1
    s.total_spikes += n_spikes
    s.get_layer(layer_id).record_tick(n_spikes, n_neurons)
end

function summary(s::DeviceTelemetryState)
    return {
        "total_ticks": s.total_ticks,
        "total_spikes": s.total_spikes,
        "error_count": s.error_count,
        "layers": {
            lid: {
                "spike_count": lt.spike_count,
                "tick_count": lt.tick_count,
                "mean_spike_rate": lt.mean_spike_rate,
                "mean_utilization": lt.mean_utilization,
            }
            for lid, lt in s.layers.items()
        },
    }
end

end # module TelemetryAccel
