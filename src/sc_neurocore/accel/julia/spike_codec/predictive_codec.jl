# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/predictive_codec

module PredictiveCodecAccel

using Statistics, LinearAlgebra

mutable struct PredictiveSpikeCodecState
    prediction_accuracy::Float64
    error_sparsity::Float64
    predictor_type::Float64
    n_channels::Float64
    alpha::Float64
    threshold::Float64
    rates::Float64
    predictor::Float64
    alpha_q8::Float64
    seed::Float64
    context_bits::Float64
    base_codec::Float64
end

function PredictiveSpikeCodecState()
    PredictiveSpikeCodecState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function predict(s::PredictiveSpikeCodecState)
    return (s.rates > s.threshold).astype(np.int8)
end

function update(s::PredictiveSpikeCodecState, actual)
    s.rates += s.alpha * (actual.astype(np.float64) - s.rates)
end

function reset(s::PredictiveSpikeCodecState)
    s.rates[:] = 0.0
end

function compress(s::PredictiveSpikeCodecState, spikes)
    import struct
    spikes = np.asarray(spikes, dtype=np.int8)
    T, N = spikes.shape
    original_bits = T * N
    if s.predictor == "world_model"
        errors, correct_predictions = predict_and_xor_world_model(
            spikes,
            N,
            history_len=s.context_bits,
            lr=s.alpha,
            threshold=s.threshold,
            seed=s.seed,
        )
        error_data, _ = s.base_codec.compress(errors)
        header = s.HEADER_MAGIC_WM + struct.pack(
            "!BdH",
            s.context_bits,
            s.alpha,
            s.seed,
        )
    elseif s.predictor == "context"
        errors, correct_predictions = _predict_and_xor_context(
            spikes,
            N,
            s.context_bits,
        )
        error_data, _ = s.base_codec.compress(errors)
        header = s.HEADER_MAGIC_CTX + struct.pack("!B", s.context_bits)
    elseif s.predictor == "lfsr"
        if _HAS_RUST:  # pragma: no cover
            flat = np.ascontiguousarray(spikes).ravel()
            err_flat, correct_predictions = _rust_predict_lfsr(
                flat,
                N,
                s.alpha_q8,
                s.seed,
            )
            errors = np.asarray(err_flat).reshape(T, N)
        else
            errors, correct_predictions = _predict_and_xor_lfsr(
                spikes,
                N,
                s.alpha_q8,
                s.seed,
            )
        error_data, _ = s.base_codec.compress(errors)
        header = s.HEADER_MAGIC_LFSR + struct.pack("!HH", s.alpha_q8, s.seed)
    else
        if _HAS_RUST:  # pragma: no cover
            flat = np.ascontiguousarray(spikes).ravel()
            err_flat, correct_predictions = _rust_predict_ema(
                flat,
                N,
                s.alpha,
                s.threshold,
            )
            errors = np.asarray(err_flat).reshape(T, N)
        else
            errors, correct_predictions = _predict_and_xor(
                spikes,
                N,
                s.alpha,
                s.threshold,
            )
        error_data, _ = s.base_codec.compress(errors)
        header = s.HEADER_MAGIC + struct.pack("!dd", s.alpha, s.threshold)
    encoded = header + error_data
    compressed_bits = length(encoded) * 8
    ratio = original_bits / max(compressed_bits, 1)
    return encoded, PredictiveCompressionResult(
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=ratio,
        n_spikes=int(sum(spikes)),
        n_neurons=N,
        n_timesteps=T,
        lossless=s.base_codec.mode == "lossless",
        prediction_accuracy=correct_predictions / max(T * N, 1),
        error_sparsity=1.0 - (int(sum(errors)) / max(T * N, 1)),
        predictor_type=s.predictor,
    )
end

function decompress(s::PredictiveSpikeCodecState, data, T, N)
    import struct
    magic = data[:4]
    if magic == s.HEADER_MAGIC_WM
        history_len = data[4]
        alpha, seed = struct.unpack("!dH", data[5:15])
        error_data = data[15:]
        errors = s.base_codec.decompress(error_data, T, N)
        return xor_and_recover_world_model(
            errors,
            N,
            history_len=history_len,
            lr=alpha,
            seed=seed,
        )
    if magic == s.HEADER_MAGIC_CTX
        context_bits = data[4]
        error_data = data[5:]
        errors = s.base_codec.decompress(error_data, T, N)
        return _xor_and_recover_context(errors, N, context_bits)
    if magic == s.HEADER_MAGIC_LFSR
        alpha_q8, seed = struct.unpack("!HH", data[4:8])
        error_data = data[8:]
        errors = s.base_codec.decompress(error_data, T, N)
        if _HAS_RUST:  # pragma: no cover
            flat = np.ascontiguousarray(errors).ravel()
            rec = np.asarray(_rust_recover_lfsr(flat, N, alpha_q8, seed))
            return rec.reshape(T, N)
        return _xor_and_recover_lfsr(errors, N, alpha_q8, seed)
    if magic == s.HEADER_MAGIC
        alpha, threshold = struct.unpack("!dd", data[4:20])
        error_data = data[20:]
        errors = s.base_codec.decompress(error_data, T, N)
        if _HAS_RUST:  # pragma: no cover
            flat = np.ascontiguousarray(errors).ravel()
            rec = np.asarray(_rust_recover_ema(flat, N, alpha, threshold))
            return rec.reshape(T, N)
        return _xor_and_recover(errors, N, alpha, threshold)
    raise ValueError(
        f"Invalid header magic: {magic!r}, expected {s.HEADER_MAGIC!r} || {s.HEADER_MAGIC_LFSR!r}"
    )
end

end # module PredictiveCodecAccel
