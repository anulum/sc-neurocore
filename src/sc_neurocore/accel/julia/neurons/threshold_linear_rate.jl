# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia threshold-linear continuous-rate transfer

module ThresholdLinearRateAccel

export step!, simulate, simulate_trace, valid, reset!, ThresholdLinearRateNeuronState

mutable struct ThresholdLinearRateNeuronState
    r::Float64
    theta::Float64
    gain::Float64
end

ThresholdLinearRateNeuronState() = ThresholdLinearRateNeuronState(0.0, 0.0, 1.0)

function valid(state::ThresholdLinearRateNeuronState)::Bool
    return all(isfinite, (state.r, state.theta, state.gain)) &&
        state.r >= 0.0 && state.gain >= 0.0
end

function step!(state::ThresholdLinearRateNeuronState, current::Float64=0.0)::Float64
    if !isfinite(current) || !valid(state)
        throw(DomainError((state.r, current), "ThresholdLinearRate contract must be finite with non-negative rate and gain"))
    end
    next_r = state.gain * max(0.0, current - state.theta)
    if !isfinite(next_r) || next_r < 0.0
        throw(DomainError(next_r, "ThresholdLinearRate output must remain finite and non-negative"))
    end
    state.r = next_r
    return next_r
end

function reset!(state::ThresholdLinearRateNeuronState)::Nothing
    state.r = 0.0
    return nothing
end

function simulate_trace(
    r::Float64,
    theta::Float64,
    gain::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0
        throw(DomainError(n_steps, "ThresholdLinearRate step count must be non-negative"))
    end
    state = ThresholdLinearRateNeuronState(r, theta, gain)
    if !valid(state) || !isfinite(current)
        throw(DomainError((r, current), "ThresholdLinearRate batch contract is invalid"))
    end
    trace = Vector{Float64}(undef, n_steps)
    for index in eachindex(trace)
        trace[index] = step!(state, current)
    end
    return (trace=trace, rf=state.r)
end

function simulate(
    n_steps::Int=1000;
    current::Float64=3.0,
    r::Float64=0.0,
    theta::Float64=0.0,
    gain::Float64=1.0,
)
    result = simulate_trace(r, theta, gain, n_steps, current)
    return result.trace, result.rf
end

end # module ThresholdLinearRateAccel
