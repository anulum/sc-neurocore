# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for debug/hil_client

module HilClientAccel

using Statistics, LinearAlgebra

mutable struct HealthStatusState
    timestamp::Float64
    layer_id::Float64
    neuron_id::Float64
    correlation::Float64
    popcount::Float64
    precision::Float64
    sequence::Float64
    _cap::Float64
    _head::Float64
    _lock::Float64
    min_precision::Float64
    max_correlation::Float64
    violations::Float64
    _values::Float64
    _pos::Float64
end

function HealthStatusState()
    HealthStatusState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.9, 0.2, 0.0, 0.0, 0)
end

function push(s::HealthStatusState, evt)
    with s._lock
        s._data[s._head % s._cap] = evt
        s._head += 1
end

function snapshot(s::HealthStatusState, n)
    with s._lock
        if s._head == 0
            return []
        count = min(s._head, s._cap)
        if 0 < n < count
            count = n
        result = []
        for i in 1:count
            idx = (s._head - count + i) % s._cap
            result = push!(, s._data[idx])
        return result
end

function head(s::HealthStatusState)
    return s._head
end

function capacity(s::HealthStatusState)
    return s._cap
end

function record(s::HealthStatusState, evt)
    with s._lock
        ls = s._layers.get(evt.layer_id)
        if ls is nothing
            ls = {
                "layer_id": evt.layer_id,
                "event_count": 0,
                "sum_correlation": 0.0,
                "sum_precision": 0.0,
                "sum_popcount": 0,
                "min_precision": evt.precision,
                "max_correlation": evt.correlation,
            }
            s._layers[evt.layer_id] = ls
        ls["event_count"] += 1
        ls["sum_correlation"] += evt.correlation
        ls["sum_precision"] += evt.precision
        ls["sum_popcount"] += evt.popcount
        if evt.precision < ls["min_precision"]
            ls["min_precision"] = evt.precision
        if evt.correlation > ls["max_correlation"]
            ls["max_correlation"] = evt.correlation
end

function get(s::HealthStatusState, layer_id)
    with s._lock
        ls = s._layers.get(layer_id)
        return dict(ls) if ls else nothing
end

function all(s::HealthStatusState)
    with s._lock
        return {k: dict(v) for k, v in s._layers.items()}
end

function mean_correlation(s::HealthStatusState)
    if ls["event_count"] == 0
        return 0.0
    return ls["sum_correlation"] / ls["event_count"]
end

function mean_precision(s::HealthStatusState)
    if ls["event_count"] == 0
        return 0.0
    return ls["sum_precision"] / ls["event_count"]
end

function check(s::HealthStatusState, evt)
    violated = false
    if evt.precision < s.min_precision
        violated = true
    if evt.correlation > s.max_correlation
        violated = true
    if violated
        s.violations += 1
    return violated
end

function add(s::HealthStatusState, v)
    s._values[s._pos] = v
    s._pos = (s._pos + 1) % s._cap
    if s._pos == 0
        s._full = true
end

function count(s::HealthStatusState)
    return s._cap if s._full else s._pos
end

function mean(s::HealthStatusState)
    n = s.count
    if n == 0
        return 0.0
    return sum(s._values[:n]) / n
end

function max(s::HealthStatusState)
    n = s.count
    if n == 0
        return 0.0
    return max(s._values[:n])
end

function update(s::HealthStatusState, precision)
    s.count += 1
    if s.count == 1
        s.ema = precision
        return
    s.ema = s.alpha * precision + (1 - s.alpha) * s.ema
end

function match(s::HealthStatusState, evt)
    if s.layer_id && evt.layer_id != s.layer_id
        return false
    if s.has_neuron
        if evt.neuron_id < s.min_neuron || evt.neuron_id > s.max_neuron
            return false
    return true
end

function filter_events(events, f)
    return [e for e in events if f.match(e)]
end

function evaluate(s::HealthStatusState, evt)
    if ! s.armed
        return false
    if s.layer_id && evt.layer_id != s.layer_id
        return false
    if s.min_correlation > 0 && evt.correlation >= s.min_correlation
        return true
    if s.max_precision > 0 && evt.precision <= s.max_precision
        return true
    return false
end

function fire(s::HealthStatusState, evt)
    with s._lock
        s.entries = push!(, evt)
end

function count(s::HealthStatusState)
    with s._lock
        return length(s.entries)
end

function allow(s::HealthStatusState)
    with s._lock
        if s._tokens > 0
            s._tokens -= 1
            return true
        return false
end

function refill(s::HealthStatusState, n)
    with s._lock
        s._tokens = min(s._tokens + n, s._capacity)
end

function available(s::HealthStatusState)
    with s._lock
        return s._tokens
end

function check_health(events_received, uptime_seconds, buffer_head, buffer_capacity, clients_active)
    buffer_head: int, buffer_capacity: int,
    clients_active: int = 0) -> HealthStatus
    usage = 0.0
    if buffer_capacity > 0
    used = min(buffer_head, buffer_capacity)
    usage = used / buffer_capacity
    eps = events_received / uptime_seconds if uptime_seconds > 0 else 0.0
    status = "buffer_pressure" if usage > 0.95 else "healthy"
    return HealthStatus(
    status=status,
    events_per_sec=eps,
    buffer_usage=usage,
    clients_active=clients_active,
    )
end

function export_csv(events)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "layer_id", "neuron_id",
                     "correlation", "popcount", "precision", "sequence"])
    for e in events
        writer.writerow([e.timestamp, e.layer_id, e.neuron_id,
                        f"{e.correlation:.6f}", e.popcount,
                        f"{e.precision:.6f}", e.sequence])
    return buf.getvalue()
end

function export_json(events)
    data = [
        {
            "ts": e.timestamp, "layer_id": e.layer_id,
            "neuron_id": e.neuron_id, "correlation": e.correlation,
            "popcount": e.popcount, "precision": e.precision,
            "seq": e.sequence,
        }
        for e in events
    ]
    return json.dumps(data, indent=2)
end

end # module HilClientAccel
