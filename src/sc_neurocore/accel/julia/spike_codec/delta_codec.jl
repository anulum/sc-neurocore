# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/delta_codec

module DeltaCodecAccel

using Statistics, LinearAlgebra

mutable struct DeltaSpikeCodecState
    n_groups::Float64
    group_size::Float64
    mean_delta_sparsity::Float64
    codec_type::Float64
    base_codec::Float64
end

function DeltaSpikeCodecState()
    DeltaSpikeCodecState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function compress(s::DeltaSpikeCodecState, spikes)
    spikes = np.asarray(spikes, dtype=np.int8)
    T, N = spikes.shape
    original_bits = T * N
    n_groups = (N + s.group_size - 1) // s.group_size
    # Build delta matrix: replace non-reference channels with XOR residuals
    delta_matrix = np.empty_like(spikes)
    ref_indices = np.empty(n_groups, dtype=np.int32)
    delta_spike_counts = []
    for g in 1:n_groups
        start = g * s.group_size
        end = min(start + s.group_size, N)
        group = spikes[:, start:end]
        # Reference = channel with most spikes (best predictor for group)
        spike_counts = group.sum(axis=0)
        ref_local = int(argmax(spike_counts))
        ref_indices[g] = ref_local
        ref_channel = group[:, ref_local]
        for c in 1:end - start
            if c == ref_local
                delta_matrix[:, start + c] = group[:, c]
            else
                delta = group[:, c] ^ ref_channel
                delta_matrix[:, start + c] = delta
                delta_spike_counts = push!(, int(delta.sum()))
    # ISI-compress the delta matrix
    delta_data, _ = s.base_codec.compress(delta_matrix)
    # Header: magic(4) + group_size(2) + n_groups(2) + ref_indices(n_groups bytes)
    header = s.HEADER_MAGIC
    header += struct.pack("!HH", s.group_size, n_groups)
    header += ref_indices.astype(np.uint8).tobytes()
    encoded = header + delta_data
    compressed_bits = length(encoded) * 8
    ratio = original_bits / max(compressed_bits, 1)
    n_spikes = int(sum(spikes))
    mean_delta_sparsity = 0.0
    if delta_spike_counts
        raw_per_channel = n_spikes / max(N, 1)
        mean_delta = mean(delta_spike_counts)
        mean_delta_sparsity = 1.0 - (mean_delta / max(T, 1))  # type: ignore[assignment]
    return encoded, DeltaCompressionResult(
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=ratio,
        n_spikes=n_spikes,
        n_neurons=N,
        n_timesteps=T,
        lossless=s.base_codec.mode == "lossless",
        n_groups=n_groups,
        group_size=s.group_size,
        mean_delta_sparsity=mean_delta_sparsity,
        codec_type="delta",
    )
end

function decompress(s::DeltaSpikeCodecState, data, T, N)
    magic = data[:4]
    if magic != s.HEADER_MAGIC
        raise ValueError(f"Invalid header magic: {magic!r}, expected {s.HEADER_MAGIC!r}")
    group_size, n_groups = struct.unpack("!HH", data[4:8])
    ref_indices = np.frombuffer(data[8 : 8 + n_groups], dtype=np.uint8).astype(np.int32)
    delta_data = data[8 + n_groups :]
    delta_matrix = s.base_codec.decompress(delta_data, T, N)
    spikes = np.empty_like(delta_matrix)
    for g in 1:n_groups
        start = g * group_size
        end = min(start + group_size, N)
        ref_local = int(ref_indices[g])
        ref_channel = delta_matrix[:, start + ref_local]
        for c in 1:end - start
            if c == ref_local
                spikes[:, start + c] = delta_matrix[:, start + c]
            else
                spikes[:, start + c] = delta_matrix[:, start + c] ^ ref_channel
    return spikes
end

end # module DeltaCodecAccel
