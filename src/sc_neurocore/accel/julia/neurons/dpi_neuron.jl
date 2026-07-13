# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia implementation of the published DPI circuit

module DpiNeuronAccel

export DPINeuronState, reset!, simulate, simulate_trace, step!, valid

mutable struct DPINeuronState
    i_mem::Float64
    i_ahp::Float64
    refractory_time::Float64
    i_threshold::Float64
    i_reset::Float64
    i_rest::Float64
    i_tau::Float64
    i_g::Float64
    i_tau_ahp::Float64
    i_ga::Float64
    i_spike::Float64
    i_0::Float64
    kappa::Float64
    alpha::Float64
    tau::Float64
    tau_ahp::Float64
    refractory_period::Float64
    dt::Float64
end

function DPINeuronState()
    return DPINeuronState(
        0.01,
        0.01,
        0.0,
        1.0,
        0.01,
        0.1,
        1.0,
        1.0,
        0.1,
        1.0,
        5.0,
        0.01,
        0.7,
        10.0,
        20.0,
        100.0,
        2.0,
        0.1,
    )
end

function valid(s::DPINeuronState)::Bool
    positive = (
        s.i_mem,
        s.i_threshold,
        s.i_reset,
        s.i_tau,
        s.i_g,
        s.i_tau_ahp,
        s.i_ga,
        s.i_spike,
        s.i_0,
        s.kappa,
        s.alpha,
        s.tau,
        s.tau_ahp,
        s.refractory_period,
        s.dt,
    )
    return all(isfinite, positive) && all(value -> value > 0.0, positive) &&
           isfinite(s.i_ahp) && s.i_ahp >= 0.0 &&
           isfinite(s.refractory_time) && s.refractory_time >= 0.0 &&
           isfinite(s.i_rest) && s.i_rest >= 0.0 &&
           s.i_reset < s.i_threshold && s.refractory_period >= s.dt
end

function _sigmoid(value::Float64)::Float64
    if value >= 0.0
        return 1.0 / (1.0 + exp(-value))
    end
    exponential = exp(value)
    return exponential / (1.0 + exponential)
end

function _feedback_current(s::DPINeuronState)::Float64
    log_current = (log(s.i_0) + s.kappa * log(s.i_mem)) / (s.kappa + 1.0)
    gate = _sigmoid(s.alpha * (s.i_mem - s.i_threshold))
    return exp(log_current) * gate
end

function step!(s::DPINeuronState, current::Float64=0.0; dt::Float64=s.dt)::Int
    total_input = s.i_rest + current
    if !valid(s) || !isfinite(current) || !isfinite(dt) || dt <= 0.0 ||
       !isfinite(total_input) || s.refractory_period < dt || total_input < 0.0
        throw(DomainError(
            (s.i_mem, s.i_ahp, s.refractory_time, dt, current),
            "DPI state/current lies outside the physical current domain",
        ))
    end

    spike_active = s.refractory_time > 0.0
    spike_current = spike_active ? s.i_spike : 0.0
    d_i_ahp = s.i_ahp / (s.tau_ahp * s.i_tau_ahp) *
              (spike_current / (1.0 + s.i_ahp / s.i_ga) - s.i_tau_ahp)
    next_i_ahp = s.i_ahp + dt * d_i_ahp

    next_i_mem = s.i_reset
    next_refractory = 0.0
    spiked = false
    if spike_active
        next_refractory = max(0.0, s.refractory_time - dt)
    else
        i_fb = _feedback_current(s)
        d_i_mem = s.i_mem / (s.tau * s.i_tau) *
                  (total_input / (1.0 + s.i_mem / s.i_g) -
                   s.i_tau + i_fb - s.i_ahp)
        next_i_mem = s.i_mem + dt * d_i_mem
        if !isfinite(next_i_mem) || next_i_mem <= 0.0
            throw(DomainError(
                next_i_mem,
                "DPI membrane Euler candidate left the physical current domain",
            ))
        end
        spiked = next_i_mem >= s.i_threshold
        if spiked
            next_i_mem = s.i_reset
            next_refractory = s.refractory_period
        end
    end

    candidates = (next_i_mem, next_i_ahp, next_refractory)
    if !all(isfinite, candidates) || next_i_mem <= 0.0 || next_i_ahp < 0.0 ||
       next_refractory < 0.0
        throw(DomainError(candidates, "DPI Euler update left the physical current domain"))
    end

    s.i_mem = next_i_mem
    s.i_ahp = next_i_ahp
    s.refractory_time = next_refractory
    s.dt = dt
    return spiked ? 1 : 0
end

function reset!(s::DPINeuronState)::Nothing
    s.i_mem = s.i_reset
    s.i_ahp = s.i_0
    s.refractory_time = 0.0
    return nothing
end

function simulate(n_steps::Int=1_000; current::Float64=5.0, dt::Float64=0.1)
    state = DPINeuronState()
    state.dt = dt
    result = simulate_trace(
        state.i_mem,
        state.i_ahp,
        state.refractory_time,
        state.i_threshold,
        state.i_reset,
        state.i_rest,
        state.i_tau,
        state.i_g,
        state.i_tau_ahp,
        state.i_ga,
        state.i_spike,
        state.i_0,
        state.kappa,
        state.alpha,
        state.tau,
        state.tau_ahp,
        state.refractory_period,
        state.dt,
        n_steps,
        current,
    )
    return result.trace, result.spikes
end

function simulate_trace(
    i_mem::Float64,
    i_ahp::Float64,
    refractory_time::Float64,
    i_threshold::Float64,
    i_reset::Float64,
    i_rest::Float64,
    i_tau::Float64,
    i_g::Float64,
    i_tau_ahp::Float64,
    i_ga::Float64,
    i_spike::Float64,
    i_0::Float64,
    kappa::Float64,
    alpha::Float64,
    tau::Float64,
    tau_ahp::Float64,
    refractory_period::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0
        throw(ArgumentError("DPI n_steps must be non-negative"))
    end
    state = DPINeuronState(
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    )
    total_input = state.i_rest + current
    if !valid(state) || !isfinite(current) || !isfinite(total_input) || total_input < 0.0
        throw(DomainError(
            (i_mem, i_ahp, refractory_time, dt, current),
            "DPI state/current lies outside the physical current domain",
        ))
    end

    trace = zeros(Float64, n_steps)
    spikes = 0
    for index in eachindex(trace)
        spikes += step!(state, current)
        trace[index] = state.i_mem
    end
    return (
        trace=trace,
        spikes=spikes,
        i_mem_f=state.i_mem,
        i_ahp_f=state.i_ahp,
        refractory_time_f=state.refractory_time,
    )
end

end # module DpiNeuronAccel
