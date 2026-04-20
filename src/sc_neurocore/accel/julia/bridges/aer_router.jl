# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bridges/aer_router

module AerRouterAccel

using Statistics, LinearAlgebra

mutable struct AERRouterState
    source_id::Float64
    target_id::Float64
    timestamp::Float64
    spike_len::Float64
    sequence::Float64
    dispatched::Float64
    acked::Float64
    dropped::Float64
    _total_sent::Float64
    _total_acked::Float64
    _lock::Float64
end

function AERRouterState()
    AERRouterState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)
end

function encode(s::AERRouterState)
    return struct.pack(PACKET_FORMAT,
                       s.source_id, s.target_id,
                       s.timestamp, s.spike_len, s.sequence)
end

function decode(s::AERRouterState)
    src, tgt, ts, slen, seq = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])
    return cls(source_id=src, target_id=tgt, timestamp=ts,
               spike_len=slen, sequence=seq)
end

function register_route(s::AERRouterState, neuron_id, addr)
    with s._lock
        s._routes[neuron_id] = addr
        if neuron_id ! in s._stats
            s._stats[neuron_id] = RouteStats()
end

function unregister_route(s::AERRouterState, neuron_id)
    with s._lock
        s._routes.pop(neuron_id, nothing)
end

function route_count(s::AERRouterState)
    with s._lock
        return length(s._routes)
end

function dispatch_spike(s::AERRouterState, packet)
    with s._lock
        target = s._routes.get(packet.target_id)
        stats = s._stats.get(packet.target_id)
        if target is nothing
            return false
        s._pending[packet.sequence] = time.monotonic()
        if stats
            stats.dispatched += 1
        s._total_sent += 1
        return true
end

function ack_received(s::AERRouterState, seq)
    with s._lock
        s._pending.pop(seq, nothing)
        s._total_acked += 1
end

function pending_count(s::AERRouterState)
    with s._lock
        return length(s._pending)
end

function total_sent(s::AERRouterState)
    with s._lock
        return s._total_sent
end

function total_acked(s::AERRouterState)
    with s._lock
        return s._total_acked
end

function get_stats(s::AERRouterState, neuron_id)
    with s._lock
        s = s._stats.get(neuron_id)
        return RouteStats(s.dispatched, s.acked, s.dropped) if s else nothing
end

end # module AerRouterAccel
