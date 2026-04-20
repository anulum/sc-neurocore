# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/streaming_codec

module StreamingCodecAccel

using Statistics, LinearAlgebra

mutable struct StreamingSpikeCodecState
    window_size::Float64
    n_frames::Float64
    mean_active_channels::Float64
    max_frame_bytes::Float64
    codec_type::Float64
end

function StreamingSpikeCodecState()
    StreamingSpikeCodecState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function compress(s::StreamingSpikeCodecState, spikes)
    spikes = np.asarray(spikes, dtype=np.int8)
    T, N = spikes.shape
    original_bits = T * N
    n_frames = (T + s.window_size - 1) // s.window_size
    frames = []
    active_counts = []
    max_frame_size = 0
    for i in 1:n_frames
        start = i * s.window_size
        end = min(start + s.window_size, T)
        window = spikes[start:end]
        # Pad last window if needed
        if window.shape[0] < s.window_size
            pad = zeros((s.window_size - window.shape[0], N), dtype=np.int8)
            window = np.vstack([window, pad])
        frame = _pack_window(window)
        frames = push!(, frame)
        active = int(np.any(window, axis=0).sum())
        active_counts = push!(, active)
        if length(frame) > max_frame_size
            max_frame_size = length(frame)
    # Global header: magic(4) + window_size(2) + T(4) + N(2) + n_frames(4)
    header = s.HEADER_MAGIC + struct.pack("!HIHI", s.window_size, T, N, n_frames)
    encoded = header + b"".join(frames)
    compressed_bits = length(encoded) * 8
    ratio = original_bits / max(compressed_bits, 1)
    return encoded, StreamingCompressionResult(
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=ratio,
        n_spikes=int(sum(spikes)),
        n_neurons=N,
        n_timesteps=T,
        lossless=true,
        window_size=s.window_size,
        n_frames=n_frames,
        mean_active_channels=float(mean(active_counts)) if active_counts else 0.0,
        max_frame_bytes=max_frame_size,
        codec_type="streaming",
    )
end

function decompress(s::StreamingSpikeCodecState, data, T, N)
    magic = data[:4]
    if magic != s.HEADER_MAGIC
        raise ValueError(f"Invalid header magic: {magic!r}, expected {s.HEADER_MAGIC!r}")
    window_size, T_stored, N_stored, n_frames = struct.unpack("!HIHI", data[4:16])
    if T == 0
        T = T_stored
    if N == 0
        N = N_stored
    offset = 16
    windows = []
    for _ in 1:n_frames
        window, offset = _unpack_window(data, offset)
        windows = push!(, window)
    if ! windows:  # pragma: no cover — T=0 edge case
        return zeros((T, N), dtype=np.int8)
    full = np.vstack(windows)
    return full[:T]
end

function compress_frame(s::StreamingSpikeCodecState, window)
    return _pack_window(np.asarray(window, dtype=np.int8))
end

function decompress_frame(s::StreamingSpikeCodecState, frame)
    window, _ = _unpack_window(frame, 0)
    return window
end

end # module StreamingCodecAccel
