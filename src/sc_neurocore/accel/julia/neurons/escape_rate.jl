# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for escape_rate

module EscapeRateAccel

export step!, simulate, simulate_trace, validate_escape_rate, EscapeRateNeuronState, reset!

mutable struct EscapeRateNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    rho_0::Float64
    delta_u::Float64
    resistance::Float64
    dt::Float64
    rng_state::UInt16
    initial_seed::UInt16
end

function EscapeRateNeuronState(seed::Integer=0xACE1)
    normalised = seed == 0 ? UInt16(0xACE1) : UInt16(seed)
    EscapeRateNeuronState(
        -70.0,
        -70.0,
        -70.0,
        -50.0,
        10.0,
        0.001,
        3.0,
        1.0,
        1.0,
        normalised,
        normalised,
    )
end

function step!(s::EscapeRateNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "EscapeRate input current must be finite"))
    end
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "EscapeRate dt must be finite and positive"))
    end

    if !validate_escape_rate(s)
        throw(DomainError(s.v, "EscapeRate state parameters must be finite and positive"))
    end

    v_inf = s.v_rest + s.resistance * I_ext
    decay = exp(-dt / s.tau_m)
    next_v = v_inf + (s.v - v_inf) * decay
    if !isfinite(v_inf) || !isfinite(decay) || !isfinite(next_v)
        throw(DomainError(next_v, "EscapeRate membrane candidate must remain finite"))
    end
    rate = s.rho_0 * safe_exp((next_v - s.v_threshold) / s.delta_u)
    hazard = rate * dt
    if !isfinite(hazard) || hazard < 0.0
        throw(DomainError(hazard, "EscapeRate hazard must remain finite and non-negative"))
    end
    p_spike = -expm1(-hazard)
    if !isfinite(p_spike) || p_spike < 0.0 || p_spike > 1.0
        throw(DomainError(p_spike, "EscapeRate spike probability must remain finite and bounded"))
    end
    sample = lfsr16_trial_sample(s.rng_state)
    threshold = probability_threshold(p_spike)
    s.rng_state = sample
    if UInt32(sample) < threshold
        s.v = s.v_reset
        return 1
    end
    s.v = next_v
    return 0
end

function validate_escape_rate(s::EscapeRateNeuronState)
    return isfinite(s.v) && isfinite(s.v_rest) && isfinite(s.v_reset) &&
           isfinite(s.v_threshold) && isfinite(s.tau_m) && s.tau_m > 0.0 &&
           isfinite(s.rho_0) && s.rho_0 > 0.0 &&
           isfinite(s.delta_u) && s.delta_u > 0.0 &&
           isfinite(s.resistance) && s.resistance > 0.0 &&
           isfinite(s.dt) && s.dt > 0.0 && s.rng_state != 0
end

function safe_exp(x::Float64)
    return exp(clamp(x, -700.0, 700.0))
end

function lfsr16_advance(state::UInt16)::UInt16
    if state == 0
        throw(DomainError(state, "EscapeRate LFSR state must be non-zero"))
    end
    value = UInt32(state)
    feedback = ((value >> 0) ⊻ (value >> 2) ⊻ (value >> 3) ⊻ (value >> 5)) & UInt32(1)
    return UInt16((value >> 1) | (feedback << 15))
end

function lfsr16_trial_sample(state::UInt16)::UInt16
    for _ in 1:8
        state = lfsr16_advance(state)
    end
    return state
end

function probability_threshold(probability::Float64)::UInt32
    if probability <= 0.0
        return UInt32(0)
    elseif probability >= 1.0
        return UInt32(65536)
    end
    return UInt32(floor(Int, probability * 65535.0) + 1)
end

function reset!(s::EscapeRateNeuronState)::Nothing
    s.v = s.v_rest
    s.rng_state = s.initial_seed
    return nothing
end

function simulate_trace(
    v::Float64,
    v_rest::Float64,
    v_reset::Float64,
    v_threshold::Float64,
    tau_m::Float64,
    rho_0::Float64,
    delta_u::Float64,
    resistance::Float64,
    dt::Float64,
    rng_state::Integer,
    n_steps::Integer,
    current::Float64,
)
    if n_steps < 0 || rng_state <= 0 || rng_state > 0xffff
        throw(DomainError(n_steps, "EscapeRate batch contract is invalid"))
    end
    seed = UInt16(rng_state)
    state = EscapeRateNeuronState(
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        rho_0,
        delta_u,
        resistance,
        dt,
        seed,
        seed,
    )
    if !validate_escape_rate(state) || !isfinite(current)
        throw(DomainError(v, "EscapeRate batch state or input is invalid"))
    end
    trace = Vector{Float64}(undef, n_steps)
    events = Vector{UInt8}(undef, n_steps)
    for index in eachindex(trace)
        event = step!(state, current)
        trace[index] = state.v
        events[index] = UInt8(event)
    end
    return (trace=trace, events=events, v_f=state.v, rng_state_f=state.rng_state)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    result = simulate_trace(
        -70.0,
        -70.0,
        -70.0,
        -50.0,
        10.0,
        0.001,
        3.0,
        1.0,
        dt,
        0xACE1,
        n_steps,
        I_ext,
    )
    return result.trace, sum(result.events)
end

end # module EscapeRateAccel
