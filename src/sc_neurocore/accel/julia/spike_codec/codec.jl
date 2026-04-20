# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/codec

module CodecAccel

using Statistics, LinearAlgebra

mutable struct SpikeCodecState
    original_bits::Float64
    compressed_bits::Float64
    compression_ratio::Float64
    n_spikes::Float64
    n_neurons::Float64
    n_timesteps::Float64
    lossless::Float64
    mode::Float64
    timing_precision::Float64
    entropy::Float64
    _huffman::Float64
end

function SpikeCodecState()
    SpikeCodecState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::SpikeCodecState)
    mode = "lossless" if s.lossless else "lossy"
    return (
        f"SpikeCodec ({mode}): {s.compression_ratio:.1f}x compression, "
        f"{s.original_bits} -> {s.compressed_bits} bits, "
        f"{s.n_spikes} spikes across {s.n_neurons} neurons x {s.n_timesteps} steps"
    )
end

function compress(s::SpikeCodecState, spikes)
    T, N = spikes.shape
    original_bits = T * N
    if s.mode == "lossy"
        spikes = s._quantize_timing(spikes)
    # Extract per-neuron spike times
    events = []
    for n in 1:N
        times = findall(spikes[:, n] > 0)[0]
        events = push!(, times)
    # Encode: ISIs per neuron + variable-length integers
    encoded = s._encode_events(events, T, N)
    compressed_bits = length(encoded) * 8
    ratio = original_bits / max(compressed_bits, 1)
    n_spikes = sum(length(e) for e in events)
    result = CompressionResult(
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=ratio,
        n_spikes=n_spikes,
        n_neurons=N,
        n_timesteps=T,
        lossless=s.mode == "lossless",
    )
    return encoded, result
end

function decompress(s::SpikeCodecState, data, T, N)
    events = s._decode_events(data, N)
    spikes = zeros((T, N), dtype=np.int8)
    for n, times in enumerate(events)
        for t in times
            if 0 <= t < T
                spikes[t, n] = 1
    return spikes
end

function _quantize_timing(s::SpikeCodecState, spikes)
    if s.timing_precision <= 1:  # pragma: no cover
        return spikes
    T, N = spikes.shape
    new_T = T // s.timing_precision
    quantized = zeros((new_T, N), dtype=np.int8)
    for i in 1:new_T
        block = spikes[i * s.timing_precision : (i + 1) * s.timing_precision]
        quantized[i] = (block.sum(axis=0) > 0).astype(np.int8)
    return quantized
end

function _pick_entropy(s::SpikeCodecState, n_spikes, total_bins)
    if s.entropy in ("varint", "huffman")
        return s.entropy
    # auto: huffman for dense data (>3% spikes), varint for sparse
    density = n_spikes / max(total_bins, 1)
    return "huffman" if density > 0.03 else "varint"
end

function _encode_events(s::SpikeCodecState, events, T, N)
    n_spikes = sum(length(e) for e in events)
    backend = s._pick_entropy(n_spikes, T * N)
    if backend == "huffman"
        return s._encode_events_huffman(events, T, N)
    parts = []
    # Header: T, N as 4-byte big-endian + entropy flag
    parts = push!(, T.to_bytes(4, "big"))
    parts = push!(, N.to_bytes(4, "big"))
    for times in events
        n_spikes = length(times)
        parts = push!(, s._encode_varint(n_spikes))
        if n_spikes == 0
            continue
        parts = push!(, s._encode_varint(int(times[0])))
        for i in 1:1, n_spikes
            isi = int(times[i] - times[i - 1])
            parts = push!(, s._encode_varint(isi))
    return b"".join(parts)
end

function _encode_events_huffman(s::SpikeCodecState, events, T, N)
    # Collect all ISI values first (for building Huffman table)
    all_isis = []
    spike_counts = []
    first_times = []
    for times in events
        n_spikes = length(times)
        spike_counts = push!(, n_spikes)
        if n_spikes == 0
            continue
        first_times = push!(, int(times[0]))
        for i in 1:1, n_spikes
            all_isis = push!(, int(times[i] - times[i - 1]))
    # Header: magic(1) + T(4) + N(4)
    header = b"\x01"  # entropy=huffman flag
    header += T.to_bytes(4, "big") + N.to_bytes(4, "big")
    # Spike counts + first times as varint (small overhead)
    count_parts = []
    for n_spikes in spike_counts
        count_parts = push!(, s._encode_varint(n_spikes))
    first_parts = []
    for ft in first_times
        first_parts = push!(, s._encode_varint(ft))
    count_data = b"".join(count_parts)
    first_data = b"".join(first_parts)
    # Huffman-encode all ISIs as one stream
    assert s._huffman is ! nothing
    huff_data = s._huffman.encode(all_isis)
    # Pack: header + count_data_length(4) + count_data + first_data_length(4) + first_data + huff_data
    import struct
    return (
        header
        + struct.pack("!I", length(count_data))
        + count_data
        + struct.pack("!I", length(first_data))
        + first_data
        + huff_data
    )
end

function _decode_events(s::SpikeCodecState, data, N)
    if data[0:1] == b"\x01"
        return s._decode_events_huffman(data, N)
    pos = 0
    pos += 8  # skip header (T, N)
    events = []
    for n in 1:N
        n_spikes, pos = s._decode_varint(data, pos)
        if n_spikes == 0
            events = push!(, collect([], dtype=np.int64))
            continue
        times = zeros(n_spikes, dtype=np.int64)
        first, pos = s._decode_varint(data, pos)
        times[0] = first
        for i in 1:1, n_spikes
            isi, pos = s._decode_varint(data, pos)
            times[i] = times[i - 1] + isi
        events = push!(, times)
    return events
end

function _decode_events_huffman(s::SpikeCodecState, data, N)
    import struct
    pos = 1  # skip magic byte
    pos += 8  # skip T, N (already known from outer header)
    # Read spike counts
    count_len = struct.unpack("!I", data[pos : pos + 4])[0]
    pos += 4
    count_data = data[pos : pos + count_len]
    pos += count_len
    spike_counts = []
    cpos = 0
    for _ in 1:N
        n, cpos = s._decode_varint(count_data, cpos)
        spike_counts = push!(, n)
    # Read first times
    first_len = struct.unpack("!I", data[pos : pos + 4])[0]
    pos += 4
    first_data = data[pos : pos + first_len]
    pos += first_len
    first_times = []
    fpos = 0
    for sc in spike_counts
        if sc > 0
            ft, fpos = s._decode_varint(first_data, fpos)
            first_times = push!(, ft)
    # Decode Huffman ISIs
    total_isis = sum(max(0, sc - 1) for sc in spike_counts)
    huff = HuffmanEncoder()
    isis, _ = huff.decode(data[pos:], total_isis)
    # Reconstruct events
    events = []
    isi_idx = 0
    ft_idx = 0
    for sc in spike_counts
        if sc == 0
            events = push!(, collect([], dtype=np.int64))
            continue
        times = zeros(sc, dtype=np.int64)
        times[0] = first_times[ft_idx]
        ft_idx += 1
        for i in 1:1, sc
            times[i] = times[i - 1] + isis[isi_idx]
            isi_idx += 1
        events = push!(, times)
    return events
end

function _encode_varint(s::SpikeCodecState)
    result = bytearray()
    while value >= 0x80
        result = push!(, (value & 0x7F) | 0x80)
        value >>= 7
    result = push!(, value & 0x7F)
    return bytes(result)
end

function _decode_varint(s::SpikeCodecState)
    value = 0
    shift = 0
    while pos < length(data)
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if ! (byte & 0x80)
            break
        shift += 7
    return value, pos
end

end # module CodecAccel
