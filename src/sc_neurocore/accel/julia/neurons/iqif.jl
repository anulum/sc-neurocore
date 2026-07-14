# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact Julia implementation of the Wu et al. 2021 IQIF soma

module IQIFAccel

export IntegerQIFNeuronState, branch_point, reset!, simulate, simulate_trace, step!, validate_iqif

const INT32_MIN = Int64(-2147483648)
const INT32_MAX = Int64(2147483647)

mutable struct IntegerQIFNeuronState
    v::Int64
    v_rest::Int64
    v_threshold::Int64
    v_reset::Int64
    a::Int64
    b::Int64
    v_max::Int64
    v_min::Int64
end

IntegerQIFNeuronState() = IntegerQIFNeuronState(128, 128, 200, 128, 1, 1, 255, 0)

in_int32(value::Int64) = INT32_MIN <= value <= INT32_MAX

function validate_iqif(state::IntegerQIFNeuronState)::Bool
    fields = (state.v, state.v_rest, state.v_threshold, state.v_reset,
              state.a, state.b, state.v_max, state.v_min)
    return all(in_int32, fields) &&
           state.a >= 0 && state.b >= 0 && state.a + state.b > 0 &&
           state.v_min < state.v_rest < state.v_threshold < state.v_max &&
           state.v_min <= state.v_reset <= state.v_max &&
           state.v_min <= state.v <= state.v_max
end

function branch_point(state::IntegerQIFNeuronState)::Int64
    numerator = state.b * state.v_threshold + state.a * state.v_rest
    return div(numerator, state.a + state.b)
end

function step!(state::IntegerQIFNeuronState, current::Integer=0)::Int
    current64 = Int64(current)
    if !validate_iqif(state) || !in_int32(current64)
        throw(DomainError(current, "IQIF requires an ordered signed-int32 contract"))
    end
    force = state.v < branch_point(state) ?
        state.a * (state.v_rest - state.v) :
        state.b * (state.v - state.v_threshold)
    candidate = state.v + (force >> 3) + current64
    if candidate > state.v_max
        state.v = state.v_reset
        return 1
    end
    state.v = max(state.v_min, candidate)
    return 0
end

function reset!(state::IntegerQIFNeuronState)::Nothing
    state.v = state.v_rest
    return nothing
end

function simulate_trace(
    v::Integer,
    v_rest::Integer,
    v_threshold::Integer,
    v_reset::Integer,
    a::Integer,
    b::Integer,
    v_max::Integer,
    v_min::Integer,
    n_steps::Integer,
    current::Integer,
)
    if n_steps < 0 || n_steps > typemax(Int32)
        throw(DomainError(n_steps, "IQIF n_steps must be in signed-int32 range"))
    end
    state = IntegerQIFNeuronState(
        Int64(v), Int64(v_rest), Int64(v_threshold), Int64(v_reset),
        Int64(a), Int64(b), Int64(v_max), Int64(v_min),
    )
    if !validate_iqif(state) || !in_int32(Int64(current))
        throw(DomainError(v, "IQIF batch contract is invalid"))
    end
    trace = Vector{Int64}(undef, n_steps)
    spikes = Int64(0)
    for index in eachindex(trace)
        spikes += step!(state, current)
        trace[index] = state.v
    end
    return (trace=trace, spikes=spikes, vf=state.v)
end

function simulate(n_steps::Int=1000; I_ext::Int=10)
    result = simulate_trace(128, 128, 200, 128, 1, 1, 255, 0, n_steps, I_ext)
    return result.trace, result.spikes
end

end # module IQIFAccel
