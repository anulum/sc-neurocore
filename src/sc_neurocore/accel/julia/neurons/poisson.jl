# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for poisson

module PoissonAccel

export step!, simulate, simulate_trace, validate_poisson, PoissonNeuronState, reset!

mutable struct PoissonNeuronState
    rate_hz::Float64
    dt_ms::Float64
    rng_state::UInt16
    initial_seed::UInt16
end

function PoissonNeuronState(seed::Integer=0xACE1)
    if seed < 0 || seed > 0xffff
        throw(DomainError(seed, "Poisson seed must be in [0, 65535]"))
    end
    normalised = seed == 0 ? UInt16(0xACE1) : UInt16(seed)
    PoissonNeuronState(100.0, 1.0, normalised, normalised)
end

function step!(s::PoissonNeuronState, I_ext::Float64=-1.0; dt::Float64=s.dt_ms)::Int
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "Poisson rate override must be finite"))
    end
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "Poisson dt_ms must be finite and positive"))
    end
    if !validate_poisson(s)
        throw(DomainError(s.rate_hz, "Poisson rate and timestep must be finite with non-negative rate and positive timestep"))
    end

    rate_hz = I_ext < 0.0 ? s.rate_hz : I_ext
    if !isfinite(rate_hz) || rate_hz < 0.0
        throw(DomainError(rate_hz, "Poisson active rate must be finite and non-negative"))
    end
    hazard = rate_hz * dt / 1000.0
    if !isfinite(hazard) || hazard < 0.0
        throw(DomainError(hazard, "Poisson interval hazard must remain finite and non-negative"))
    end
    p_spike = -expm1(-hazard)
    if !isfinite(p_spike) || p_spike < 0.0 || p_spike > 1.0
        throw(DomainError(p_spike, "Poisson spike probability must remain finite and bounded"))
    end
    sample = lfsr16_trial_sample(s.rng_state)
    threshold = probability_threshold(p_spike)
    s.rng_state = sample
    return UInt32(sample) < threshold ? 1 : 0
end

function validate_poisson(s::PoissonNeuronState)
    return isfinite(s.rate_hz) && s.rate_hz >= 0.0 &&
           isfinite(s.dt_ms) && s.dt_ms > 0.0 &&
           s.rng_state != 0 && s.initial_seed != 0
end

function lfsr16_advance(state::UInt16)::UInt16
    if state == 0
        throw(DomainError(state, "Poisson LFSR state must be non-zero"))
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

function reset!(s::PoissonNeuronState)::Nothing
    s.rng_state = s.initial_seed
    return nothing
end

function simulate_trace(
    rate_hz::Float64,
    dt_ms::Float64,
    rng_state::Integer,
    n_steps::Integer,
    rate_override::Float64,
)
    if n_steps < 0 || rng_state <= 0 || rng_state > 0xffff
        throw(DomainError(n_steps, "Poisson batch contract is invalid"))
    end
    seed = UInt16(rng_state)
    state = PoissonNeuronState(rate_hz, dt_ms, seed, seed)
    if !validate_poisson(state) || !isfinite(rate_override)
        throw(DomainError(rate_hz, "Poisson batch state or rate override is invalid"))
    end
    events = Vector{UInt8}(undef, n_steps)
    for index in eachindex(events)
        events[index] = UInt8(step!(state, rate_override))
    end
    return (events=events, rng_state_f=state.rng_state)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    result = simulate_trace(100.0, dt, 0xACE1, n_steps, I_ext)
    return result.events, sum(result.events)
end

end # module PoissonAccel
