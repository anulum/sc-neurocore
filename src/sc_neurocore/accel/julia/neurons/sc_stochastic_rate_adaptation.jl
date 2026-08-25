# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for SC stochastic rate adaptation

module SCStochasticRateAdaptationAccel

export step!, simulate, SCStochasticRateAdaptationNeuronState, valid, reset!, f_onset, rk4_candidate

mutable struct SCStochasticRateAdaptationNeuronState
    a::Float64
    f_max::Float64
    beta::Float64
    i_half::Float64
    tau_a::Float64
    delta_a::Float64
    dt::Float64
    rng_threshold::Float64
end

function SCStochasticRateAdaptationNeuronState()
    SCStochasticRateAdaptationNeuronState(0.0, 200.0, 0.1, 5.0, 100.0, 0.5, 1.0, 0.0)
end

function valid(s::SCStochasticRateAdaptationNeuronState)::Bool
    return isfinite(s.a) && s.a >= 0.0 &&
        isfinite(s.f_max) && s.f_max > 0.0 &&
        isfinite(s.beta) && s.beta > 0.0 &&
        isfinite(s.i_half) &&
        isfinite(s.tau_a) && s.tau_a > 0.0 &&
        isfinite(s.delta_a) && s.delta_a >= 0.0 &&
        isfinite(s.dt) && s.dt > 0.0 &&
        isfinite(s.rng_threshold) && 0.0 <= s.rng_threshold < 1.0
end

function f_onset(s::SCStochasticRateAdaptationNeuronState, x::Float64)::Float64
    z = s.beta * (x - s.i_half)
    if z == Inf
        return s.f_max
    elseif z == -Inf
        return 0.0
    elseif !isfinite(z)
        return NaN
    end
    if z >= 0.0
        return s.f_max / (1.0 + exp(-z))
    end
    exp_z = exp(z)
    return s.f_max * exp_z / (1.0 + exp_z)
end

function adaptation_rhs(s::SCStochasticRateAdaptationNeuronState, a::Float64, I_ext::Float64)
    if !isfinite(a) || a < 0.0
        return 0.0, 0.0, false
    end
    rate = f_onset(s, I_ext - a)
    if !isfinite(rate) || rate < 0.0 || rate > s.f_max
        return 0.0, 0.0, false
    end
    return -a / s.tau_a + s.delta_a * rate, rate, true
end

function rk4_candidate(s::SCStochasticRateAdaptationNeuronState, I_ext::Float64, dt::Float64=s.dt)
    k1, r1, ok = adaptation_rhs(s, s.a, I_ext)
    ok || return 0.0, 0.0, false
    k2, r2, ok = adaptation_rhs(s, s.a + 0.5 * dt * k1, I_ext)
    ok || return 0.0, 0.0, false
    k3, r3, ok = adaptation_rhs(s, s.a + 0.5 * dt * k2, I_ext)
    ok || return 0.0, 0.0, false
    k4, r4, ok = adaptation_rhs(s, s.a + dt * k3, I_ext)
    ok || return 0.0, 0.0, false

    next_a = s.a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    hazard = average_rate * dt / 1000.0
    if !isfinite(next_a) || next_a < 0.0 || !isfinite(hazard) || hazard < 0.0
        return 0.0, 0.0, false
    end
    probability = -expm1(-hazard)
    if !isfinite(probability) || probability < 0.0 || probability > 1.0
        return 0.0, 0.0, false
    end
    return next_a, probability, true
end

function step!(s::SCStochasticRateAdaptationNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        return 0
    end

    next_a, probability, ok = rk4_candidate(s, I_ext, dt)
    if !ok
        return 0
    end

    s.dt = dt
    s.a = next_a
    return s.rng_threshold < probability ? 1 : 0
end

function reset!(s::SCStochasticRateAdaptationNeuronState)::Nothing
    s.a = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = SCStochasticRateAdaptationNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.a
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

"""Execute a controlled current/uniform trace for cross-runtime parity."""
function simulate_controlled(currents::AbstractVector{Float64}, uniforms::AbstractVector{Float64}; state::SCStochasticRateAdaptationNeuronState=SCStochasticRateAdaptationNeuronState())
    length(currents) == length(uniforms) || error("current/uniform length mismatch")
    adaptation = Vector{Float64}(undef, length(currents))
    events = Vector{Int64}(undef, length(currents))
    for index in eachindex(currents)
        state.rng_threshold = uniforms[index]
        events[index] = step!(state, currents[index])
        adaptation[index] = state.a
    end
    (; adaptation, events, state)
end

end # module SCStochasticRateAdaptationAccel
