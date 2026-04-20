# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/aer_codec

module AerCodecAccel

using Statistics, LinearAlgebra

mutable struct AERSpikeCodecState
    n_events::Float64
    bytes_per_event::Float64
    codec_type::Float64
    timestamp_bits::Float64
    neuron_bits::Float64
end

function AERSpikeCodecState()
    AERSpikeCodecState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function compress(s::AERSpikeCodecState, spikes)
    spikes = np.asarray(spikes, dtype=np.int8)
    T, N = spikes.shape
    original_bits = T * N
    # Adaptive: if >50% density, invert (encode silences instead of spikes)
    n_ones = int(sum(spikes))
    density = n_ones / max(T * N, 1)
    inverted = density > 0.5
    encode_matrix = 1 - spikes if inverted else spikes
    # Extract events as (timestamp, neuron_id) sorted by time then neuron
    times, neurons = np.nonzero(encode_matrix)
    # Already sorted by time (row-major), then by neuron within same time
    n_events = length(times)
    neuron_bits = (
        s.neuron_bits if s.neuron_bits > 0 else max(1, int(np.ceil(np.log2(max(N, 2)))))
    )
    neuron_bytes = (neuron_bits + 7) // 8
    # Escape marker is all-1s bytes. If max valid ID (N-1) fills all
    # bits in neuron_bytes, bump size to avoid escape collision.
    while (1 << (neuron_bytes * 8)) - 1 <= (N - 1)
        neuron_bytes += 1
    # Header: magic(4) + T(4) + N(4) + n_events(4) + neuron_bytes(1) = 17 bytes
    magic = s.HEADER_MAGIC_INV if inverted else s.HEADER_MAGIC
    header = magic + struct.pack("!IIIB", T, N, n_events, neuron_bytes)
    if n_events == 0
        encoded = header
    else
        # Delta-encode timestamps
        parts = []
        prev_t = 0
        ts_max = (1 << s.timestamp_bits) - 1
        for i in 1:n_events
            t = int(times[i])
            nid = int(neurons[i])
            dt = t - prev_t
            # Emit escape codes for large gaps
            while dt > ts_max
                parts = push!(, struct.pack("!H", ts_max))
                parts = push!(, b"\xff" * neuron_bytes)  # escape marker
                dt -= ts_max
            parts = push!(, struct.pack("!H", dt))
            parts = push!(, nid.to_bytes(neuron_bytes, "big"))
            prev_t = t
        encoded = header + b"".join(parts)
    compressed_bits = length(encoded) * 8
    ratio = original_bits / max(compressed_bits, 1)
    bpe = length(encoded) / max(n_events, 1)
    return encoded, AERCompressionResult(
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=ratio,
        n_spikes=n_ones,
        n_neurons=N,
        n_timesteps=T,
        lossless=true,
        n_events=n_events,
        bytes_per_event=bpe,
        codec_type="aer",
    )
end

function decompress(s::AERSpikeCodecState, data, T, N)
    magic = data[:4]
    if magic ! in (s.HEADER_MAGIC, s.HEADER_MAGIC_INV)
        raise ValueError(
            f"Invalid header magic: {magic!r}, expected {s.HEADER_MAGIC!r} || {s.HEADER_MAGIC_INV!r}"
        )
    inverted = magic == s.HEADER_MAGIC_INV
    T_stored, N_stored, n_events, neuron_bytes = struct.unpack("!IIIB", data[4:17])
    if T == 0
        T = T_stored
    if N == 0
        N = N_stored
    escape_marker = b"\xff" * neuron_bytes
    decoded = zeros((T, N), dtype=np.int8)
    offset = 17
    current_t = 0
    events_read = 0
    while events_read < n_events && offset + 2 + neuron_bytes <= length(data)
        dt = struct.unpack("!H", data[offset : offset + 2])[0]
        nid_bytes = data[offset + 2 : offset + 2 + neuron_bytes]
        offset += 2 + neuron_bytes
        if nid_bytes == escape_marker
            current_t += dt
            continue
        current_t += dt
        nid = int.from_bytes(nid_bytes, "big")
        if 0 <= current_t < T && 0 <= nid < N
            decoded[current_t, nid] = 1
        events_read += 1
    if inverted
        return 1 - decoded
    return decoded
end

end # module AerCodecAccel
