# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for adex

module AdexAccel

export step!, simulate, simulate_trace, simulate_complete, reset!, AdExNeuronState

mutable struct AdExNeuronState
    v::Float64
    w::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    v_rh::Float64
    delta_t::Float64
    tau::Float64
    tau_w::Float64
    a::Float64
    b::Float64
    c_m::Float64
    dt::Float64
end

function AdExNeuronState()
    AdExNeuronState(-65.0, 0.0, -65.0, -68.0, -50.0, -55.0, 2.0, 20.0, 100.0, 0.5, 7.0, 200.0, 0.1)
end

function step!(s::AdExNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !all(isfinite, (s.v, s.w, s.v_rest, s.v_reset, s.v_threshold, s.v_rh, s.delta_t, s.tau, s.tau_w, s.a, s.b, s.c_m, dt))
        throw(DomainError(s.v, "AdEx state parameters must be finite"))
    end
    if s.delta_t <= 0.0 || s.tau <= 0.0 || s.tau_w <= 0.0 || s.c_m <= 0.0 || dt <= 0.0
        throw(DomainError(s.delta_t, "AdEx delta_t, tau, tau_w, c_m, and dt must be positive"))
    end
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "AdEx input current must be finite"))
    end

    exp_term = s.delta_t * exp(clamp((s.v - s.v_rh) / s.delta_t, -20.0, 20.0))
    dv = ((-(s.v - s.v_rest) + exp_term) / s.tau + (-s.w + I_ext) / s.c_m) * dt
    dw = (s.a * (s.v - s.v_rest) - s.w) / s.tau_w * dt
    next_v = s.v + dv
    next_w = s.w + dw
    if !all(isfinite, (exp_term, dv, dw, next_v, next_w))
        throw(DomainError(next_v, "AdEx integrator update must remain finite"))
    end
    if next_v >= s.v_threshold
        spike_w = next_w + s.b
        if !isfinite(spike_w)
            throw(DomainError(spike_w, "AdEx spike adaptation update must remain finite"))
        end
        s.v = s.v_reset
        s.w = spike_w
        s.dt = dt
        return 1
    end
    s.v = next_v
    s.w = next_w
    s.dt = dt
    return 0
end

"""Restore the dynamic state while preserving configured parameters."""
function reset!(s::AdExNeuronState)
    s.v = s.v_rest
    s.w = 0.0
    return nothing
end

"""Run the full maintained contract and return both state traces plus events."""
function simulate_complete(
    v::Float64,
    w::Float64,
    v_rest::Float64,
    v_reset::Float64,
    v_threshold::Float64,
    v_rh::Float64,
    delta_t::Float64,
    tau::Float64,
    tau_w::Float64,
    a::Float64,
    b::Float64,
    c_m::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(DomainError(n_steps, "AdEx n_steps must be non-negative"))
    end
    s = AdExNeuronState(
        v,
        w,
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        tau_w,
        a,
        b,
        c_m,
        dt,
    )
    v_trace = zeros(n_steps)
    w_trace = zeros(n_steps)
    events = zeros(UInt8, n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        v_trace[t] = s.v
        w_trace[t] = s.w
        events[t] = UInt8(result)
        if result > 0
            spikes += 1
        end
    end
    return (
        v_trace=v_trace,
        w_trace=w_trace,
        events=events,
        spikes=spikes,
        vf=s.v,
        wf=s.w,
    )
end

"""Run the compatibility voltage-trace surface."""
function simulate_trace(
    v::Float64,
    w::Float64,
    v_rest::Float64,
    v_reset::Float64,
    v_threshold::Float64,
    v_rh::Float64,
    delta_t::Float64,
    tau::Float64,
    tau_w::Float64,
    a::Float64,
    b::Float64,
    c_m::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    result = simulate_complete(
        v, w, v_rest, v_reset, v_threshold, v_rh, delta_t,
        tau, tau_w, a, b, c_m, dt, n_steps, I_ext,
    )
    return (trace=result.v_trace, spikes=result.spikes, vf=result.vf, wf=result.wf)
end

"""Run the default AdEx neuron under a constant current."""
function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    result = simulate_trace(
        -65.0,
        0.0,
        -65.0,
        -68.0,
        -50.0,
        -55.0,
        2.0,
        20.0,
        100.0,
        0.5,
        7.0,
        200.0,
        dt,
        n_steps,
        I_ext,
    )
    return result.trace, result.spikes
end

end # module AdexAccel
