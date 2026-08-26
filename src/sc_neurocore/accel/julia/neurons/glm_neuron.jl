# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for GLMNeuron

module GlmNeuronAccel

export step!, simulate, reset!, valid, GLMNeuronState

mutable struct GLMNeuronState
    mu::Float64
    dt_ms::Float64
    k::Vector{Float64}
    h::Vector{Float64}
    stim_buf::Vector{Float64}
    spike_buf::Vector{Float64}
end

function GLMNeuronState(n_k::Int=10, n_h::Int=20)
    (n_k >= 1 && n_h >= 1) || throw(ArgumentError("n_k and n_h must be at least 1"))
    k = [exp(-(i - 1) / 3.0) * 0.5 for i in 1:n_k]
    h = [-5.0 * exp(-(t - 1) / 2.0) + 0.5 * exp(-(t - 1) / 10.0) for t in 1:n_h]
    GLMNeuronState(-3.0, 1.0, k, h, zeros(n_k), zeros(n_h))
end

function valid(s::GLMNeuronState)
    isfinite(s.mu) &&
        isfinite(s.dt_ms) && 0.0 < s.dt_ms <= 1000.0 &&
        length(s.k) == length(s.stim_buf) &&
        length(s.h) == length(s.spike_buf) &&
        !isempty(s.k) && !isempty(s.h) &&
        all(isfinite, s.k) && all(isfinite, s.h) &&
        all(isfinite, s.stim_buf) && all(isfinite, s.spike_buf)
end

function step!(s::GLMNeuronState, stimulus::Float64, uniform::Float64)
    isfinite(stimulus) || throw(ArgumentError("stimulus must be finite"))
    (isfinite(uniform) && 0.0 <= uniform < 1.0) ||
        throw(ArgumentError("uniform must be finite and within [0, 1)"))
    valid(s) || throw(ArgumentError("GLM state and parameters must satisfy the public bounds"))

    stim_candidate = circshift(s.stim_buf, 1)
    stim_candidate[1] = stimulus
    log_rate = sum(s.k .* stim_candidate) + sum(s.h .* s.spike_buf) + s.mu
    log_rate = clamp(log_rate, -20.0, 20.0)
    p = exp(log_rate) * s.dt_ms / 1000.0
    spike = uniform < min(p, 1.0) ? 1 : 0
    spike_candidate = circshift(s.spike_buf, 1)
    spike_candidate[1] = Float64(spike)

    s.stim_buf = stim_candidate
    s.spike_buf = spike_candidate
    spike
end

function reset!(s::GLMNeuronState)
    fill!(s.stim_buf, 0.0)
    fill!(s.spike_buf, 0.0)
    nothing
end

function simulate(n_steps::Int=1000; stimulus::Float64=5.0, seed::Int=42)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = GLMNeuronState()
    state = UInt64(seed == 0 ? 1 : seed)
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        # xorshift64* uniform sampling — service-local regression evidence
        # only, not a cross-backend parity surface.
        state = state ⊻ (state >> 12)
        state = state ⊻ (state << 25)
        state = state ⊻ (state >> 27)
        sample = Float64((state * 0x2545F4914F6CDD1D) >> 11) / 9007199254740992.0
        spikes += step!(s, stimulus, sample)
        trace[t] = s.spike_buf[1]
    end
    trace, spikes
end

end # module GlmNeuronAccel
