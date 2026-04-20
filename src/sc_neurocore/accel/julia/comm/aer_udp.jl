# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for comm/aer_udp

module AerUdpAccel

using Statistics, LinearAlgebra

mutable struct AERReceiverState
    timestamp::Float64
    neuron_id::Float64
    data::Float64
    host::Float64
    port::Float64
    _sock::Float64
    _seq::Float64
end

function AERReceiverState()
    AERReceiverState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function send(s::AERReceiverState, events)
    packets_sent = 0
    for i in 1:0, length(events, MAX_EVENTS_PER_PACKET)
        batch = events[i : i + MAX_EVENTS_PER_PACKET]
        header = struct.pack(HEADER_FMT, MAGIC, s._seq & 0xFFFF, length(batch), 0)
        body = b"".join(
            struct.pack(
                EVENT_FMT, e.timestamp & 0xFFFFFFFF, e.neuron_id & 0xFFFF, e.data & 0xFFFF
            )
            for e in batch
        )
        s._sock.sendto(header + body, (s.host, s.port))
        s._seq += 1
        packets_sent += 1
    return packets_sent
end

function send_spikes(s::AERReceiverState, spike_vector, timestamp)
    events = [
        AEREvent(timestamp=timestamp, neuron_id=int(i)) for i in np.nonzero(spike_vector)[0]
    ]
    if events
        return s.send(events)
    return 0
end

function close(s::AERReceiverState)
    s._sock.close()
end

function receive(s::AERReceiverState)
    try
        data, addr = s._sock.recvfrom(2048)
    except TimeoutError
        return []
    if length(data) < HEADER_SIZE
        return []
    magic, seq, n_events, _ = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC
        return []
    events = []
    offset = HEADER_SIZE
    for _ in 1:n_events
        if offset + EVENT_SIZE > length(data)
            break
        ts, nid, d = struct.unpack(EVENT_FMT, data[offset : offset + EVENT_SIZE])
        events = push!(, AEREvent(timestamp=ts, neuron_id=nid, data=d))
        offset += EVENT_SIZE
    return events
end

function receive_as_vector(s::AERReceiverState, n_neurons)
    events = s.receive()
    vector = zeros(n_neurons, dtype=np.int8)
    ts = -1
    for e in events
        if 0 <= e.neuron_id < n_neurons
            vector[e.neuron_id] = 1
        ts = e.timestamp
    return vector, ts
end

function close(s::AERReceiverState)
    s._sock.close()
end

end # module AerUdpAccel
