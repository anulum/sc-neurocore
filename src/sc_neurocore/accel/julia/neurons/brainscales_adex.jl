# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for brainscales_adex

module BrainscalesAdexAccel

export step!, simulate, BrainScaleSAdExNeuronState

mutable struct BrainScaleSAdExNeuronState
    v::Float64
    w::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    delta_t::Float64
    v_rh::Float64
    tau::Float64
    tau_w::Float64
    a::Float64
    b::Float64
    hw_speedup::Float64
    dt::Float64
end

function BrainScaleSAdExNeuronState()
    BrainScaleSAdExNeuronState(-65.0, 0.0, -65.0, -68.0, -50.0, 2.0, -55.0, 20.0, 100.0, 0.5, 7.0, 1000.0, 0.1)
end

function step!(s::BrainScaleSAdExNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    s.dt = dt
    if !all(isfinite, (s.v, s.w, s.v_rest, s.v_reset, s.v_threshold, s.delta_t, s.v_rh, s.tau, s.tau_w, s.a, s.b, s.hw_speedup, s.dt))
        throw(DomainError(s.v, "BrainScaleS AdEx state parameters must be finite"))
    end
    if s.delta_t <= 0.0 || s.tau <= 0.0 || s.tau_w <= 0.0 || s.hw_speedup <= 0.0 || s.dt <= 0.0
        throw(DomainError(s.delta_t, "BrainScaleS AdEx delta_t, tau, tau_w, hw_speedup, and dt must be positive"))
    end
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "BrainScaleS AdEx input current must be finite"))
    end

    dt_hw = s.dt * s.hw_speedup
    dt_bio = dt_hw / s.hw_speedup
    exp_arg = clamp((s.v - s.v_rh) / s.delta_t, -20.0, 20.0)
    exp_term = s.delta_t * exp(exp_arg)
    dv = (-(s.v - s.v_rest) + exp_term - s.w + I_ext) / s.tau * dt_bio
    dw = (s.a * (s.v - s.v_rest) - s.w) / s.tau_w * dt_bio
    next_v = s.v + dv
    next_w = s.w + dw
    if !all(isfinite, (dt_hw, dt_bio, exp_term, dv, dw, next_v, next_w))
        throw(DomainError(next_v, "BrainScaleS AdEx integrator update must remain finite"))
    end
    if next_v >= s.v_threshold
        spike_w = next_w + s.b
        if !isfinite(spike_w)
            throw(DomainError(spike_w, "BrainScaleS AdEx spike adaptation update must remain finite"))
        end
        s.v = s.v_reset
        s.w = spike_w
        return 1
    end
    s.v = next_v
    s.w = next_w
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BrainScaleSAdExNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module BrainscalesAdexAccel
