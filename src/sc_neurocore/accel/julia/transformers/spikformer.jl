# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for transformers/spikformer

module SpikformerAccel

using Statistics, LinearAlgebra

mutable struct CPGPositionalEncodingState
    embed_dim::Float64
    num_heads::Float64
    T::Float64
    threshold::Float64
    d_model::Float64
    d_state::Float64
    dt::Float64
    max_len::Float64
end

function CPGPositionalEncodingState()
    CPGPositionalEncodingState(0.0, 1.0, 8.0, 1.0, 0.0, 64.0, 0.01, 1024.0)
end

function _spike_fn(s::CPGPositionalEncodingState, membrane)
    spikes = (membrane >= s.threshold).astype(np.float64)
    membrane = membrane - spikes * s.threshold
    return spikes, membrane
end

function forward(s::CPGPositionalEncodingState, x)
    squeeze = x.ndim == 1
    if squeeze
        x = x[np.newaxis]
    seq_len = x.shape[0]
    # Linear projections
    Q_proj = x @ s.W_q
    K_proj = x @ s.W_k
    V_proj = x @ s.W_v
    # Accumulate over T timesteps with spike-driven attention
    output_acc = np.zeros_like(x)
    s._v_q = np.zeros_like(Q_proj)
    s._v_k = np.zeros_like(K_proj)
    for t in 1:s.T
        # Rate-code input: spike with probability proportional to projection
        s._v_q += clamp(Q_proj, 0, nothing) / s.T
        s._v_k += clamp(K_proj, 0, nothing) / s.T
        Q_spikes, s._v_q = s._spike_fn(s._v_q)
        K_spikes, s._v_k = s._spike_fn(s._v_k)
        # SSA: spike AND instead of softmax
        # attn_weights[i,j] = Q_spikes[i] AND K_spikes[j] (dot product of binary)
        attn = Q_spikes @ K_spikes.T  # (seq, seq) — counts of matching spikes
        scale = max(sqrt(s.head_dim), 1.0)
        attn = attn / scale
        # Weighted sum of V
        output_acc += attn @ V_proj
    output = (output_acc / s.T) @ s.W_out
    if squeeze
        output = output[0]
    return output
end

function num_multiply_ops(s::CPGPositionalEncodingState)
    return 0
end

function reset(s::CPGPositionalEncodingState)
    s._h = zeros(s.d_state)
    s._v = zeros(s.d_model)
end

function step(s::CPGPositionalEncodingState, x)
    s._h = s.A * s._h + s.B @ x
    y = s.C @ s._h
    s._v += y
    spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= spikes * s.threshold
    return spikes, y
end

function forward(s::CPGPositionalEncodingState, x_seq)
    s.reset()
    T = x_seq.shape[0]
    out = np.zeros_like(x_seq)
    for t in 1:T
        spikes, _ = s.step(x_seq[t])
        out[t] = spikes
    return out
end

function encode(s::CPGPositionalEncodingState, seq_len)
    t = collect(seq_len)[:, np.newaxis]
    angles = t * s.frequencies[np.newaxis, :] * 0.01 + s.phases[np.newaxis, :]
    return (sin(angles) + 1.0) / 2.0
end

function encode_spikes(s::CPGPositionalEncodingState, seq_len, rng)
    if rng is nothing
        rng = np.random.RandomState(0)
    rates = s.encode(seq_len)
    return (rng.random(rates.shape) < rates).astype(np.int8)
end

end # module SpikformerAccel
