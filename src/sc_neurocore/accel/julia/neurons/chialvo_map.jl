# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Checked Julia accelerator for the Chialvo map

module ChialvoMapAccel

export ChialvoMapNeuronState, reset!, simulate, simulate_trace, step!, validate

"""State and parameters for Chialvo's two-dimensional discrete map."""
mutable struct ChialvoMapNeuronState
    x::Float64
    y::Float64
    a::Float64
    b::Float64
    c::Float64
    k::Float64
    x_threshold::Float64
end

ChialvoMapNeuronState() = ChialvoMapNeuronState(0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0)

"""Return whether every state and parameter field is finite."""
function validate(state::ChialvoMapNeuronState)::Bool
    return isfinite(state.x) &&
           isfinite(state.y) &&
           isfinite(state.a) &&
           isfinite(state.b) &&
           isfinite(state.c) &&
           isfinite(state.k) &&
           isfinite(state.x_threshold)
end

safe_exp(value::Float64)::Float64 = exp(clamp(value, -500.0, 500.0))

"""Advance one simultaneous map iteration under an additive perturbation."""
function step!(state::ChialvoMapNeuronState, current::Float64 = 0.0)::Int64
    validate(state) || throw(ArgumentError("invalid Chialvo map runtime state"))
    isfinite(current) || throw(ArgumentError("invalid Chialvo map current"))

    x_previous = state.x
    x_squared = state.x * state.x
    exponential = safe_exp(state.y - state.x)
    x_next = x_squared * exponential + state.k + current
    y_next = state.a * state.y - state.b * state.x + state.c
    if !isfinite(x_next) || !isfinite(y_next)
        throw(OverflowError("invalid Chialvo map candidate state"))
    end

    state.x = x_next
    state.y = y_next
    return x_previous < state.x_threshold <= state.x ? Int64(1) : Int64(0)
end

"""Run checked iterations and return trace, event count, and final state."""
function simulate_trace(
    x0::Float64,
    y0::Float64,
    a::Float64,
    b::Float64,
    c::Float64,
    k::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    state = ChialvoMapNeuronState(x0, y0, a, b, c, k, x_threshold)
    validate(state) || throw(ArgumentError("invalid Chialvo map configuration"))
    trace = Vector{Float64}(undef, n_steps)
    spikes = Int64(0)
    for index in eachindex(trace)
        spikes += step!(state, current)
        trace[index] = state.x
    end
    return (; trace, spikes, xf = state.x, yf = state.y)
end

"""Run the default Chialvo parameter set for compatibility callers."""
function simulate(n_steps::Int = 1000; I_ext::Float64 = 0.0)
    result = simulate_trace(0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, n_steps, I_ext)
    return result.trace, result.spikes
end

"""Restore state variables while preserving configured parameters."""
function reset!(state::ChialvoMapNeuronState)::Nothing
    state.x = 0.0
    state.y = 0.0
    return nothing
end

end # module ChialvoMapAccel
