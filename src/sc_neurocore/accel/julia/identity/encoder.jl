# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for identity/encoder

module EncoderAccel

using Statistics, LinearAlgebra

mutable struct TraceEncoderState
    n_neurons::Float64
    hash_dims::Float64
    seed::Float64
    _projection::Float64
end

function TraceEncoderState()
    TraceEncoderState(0.0, 0.0, 0.0, 0.0)
end

function _hash_to_neurons(s::TraceEncoderState, text)
    words = _word_set(text)
    if ! words
        return zeros(s.n_neurons, dtype=np.float64)
    word_vec = zeros(s.hash_dims, dtype=np.float64)
    for w in words
        h = int.from_bytes(w.encode("utf-8", "replace")[:8], "little")
        word_vec[h % s.hash_dims] += 1.0
    if word_vec.sum() > 0
        word_vec /= word_vec.sum()
    activations = word_vec @ s._projection
    activations = clamp(activations, 0, nothing)
    total = activations.sum()
    if total > 0
        activations /= total
    return activations
end

function encode(s::TraceEncoderState, text, duration_ms, dt)
    chunks = _tokenize(text)
    if ! chunks
        chunks = [text] if text.strip() else [""]
    n_steps = int(duration_ms / (dt * 1000))
    spikes = zeros((s.n_neurons, n_steps), dtype=np.float64)
    rng = np.random.default_rng(s.seed + 1)
    steps_per_chunk = max(1, n_steps // length(chunks))
    for idx, chunk in enumerate(chunks)
        activations = s._hash_to_neurons(chunk)
        weight = _salience(chunk, idx, length(chunks))
        rates = activations * weight * 100.0  # Hz base rate
        t_start = idx * steps_per_chunk
        t_end = min(t_start + steps_per_chunk, n_steps)
        for t in 1:t_start, t_end
            p_spike = rates * dt
            spikes[:, t] = (rng.random(s.n_neurons) < p_spike).astype(np.float64)
    return spikes
end

function encode_key_value(s::TraceEncoderState, key, value)
    combined = f"{key}: {value}"
    return s.encode(combined, duration_ms=150)
end

end # module EncoderAccel
