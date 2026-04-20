# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hybrid_linear_attention

module HybridLinearAttentionAccel

export step!, simulate, HybridLinearAttentionNeuronState

mutable struct HybridLinearAttentionNeuronState
    dim::Float64
    lambda_decay::Float64
    window_size::Float64
    dt::Float64
    v::Float64
    _state_kv::Float64
    _window_buf::Float64
    _window_idx::Float64
end

function HybridLinearAttentionNeuronState()
    HybridLinearAttentionNeuronState(16.0, 0.95, 16.0, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function _phi(s::HybridLinearAttentionNeuronState, x)
    return (x > 0.0) ? x + 1.0 : exp(x)
end

function step_qkv(s::HybridLinearAttentionNeuronState, query, key, value)
    phi_q = s._phi(query)
    phi_k = s._phi(key)
    for i in 1:s.dim
        s._state_kv[i] *= s.lambda_decay
    end
    idx = Int(abs(phi_k) * s.dim) % s.dim
    s._state_kv[idx] += phi_k * value
    global_out = phi_q * s._state_kv[idx]
    s._window_buf[s._window_idx % s.window_size] = value
    s._window_idx += 1
    local_out = sum(s._window_buf) / s.window_size
    s.v = 0.5 * global_out + 0.5 * local_out
    return s.v
end

function step!(s::HybridLinearAttentionNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        out = s.step_qkv(I_ext, I_ext, I_ext)
        return (out > 1.0) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HybridLinearAttentionNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.dim
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module HybridLinearAttentionAccel
