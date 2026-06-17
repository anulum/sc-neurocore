# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for escape_rate

module EscapeRateAccel

export step!, simulate, validate_escape_rate, EscapeRateNeuronState, reset!

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
end

function EscapeRateNeuronState()
    EscapeRateNeuronState(-70.0, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0, 1.0)
end

function step!(s::EscapeRateNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "EscapeRate input current must be finite"))
    end
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "EscapeRate dt must be finite and positive"))
    end

    previous_dt = s.dt
    s.dt = dt
    if !validate_escape_rate(s)
        s.dt = previous_dt
        throw(DomainError(s.v, "EscapeRate state parameters must be finite and positive"))
    end

    v_inf = s.v_rest + s.resistance * I_ext
    decay = exp(-s.dt / s.tau_m)
    next_v = v_inf + (s.v - v_inf) * decay
    if !isfinite(v_inf) || !isfinite(decay) || !isfinite(next_v)
        throw(DomainError(next_v, "EscapeRate membrane candidate must remain finite"))
    end
    rate = s.rho_0 * safe_exp((next_v - s.v_threshold) / s.delta_u)
    hazard = rate * s.dt
    if !isfinite(hazard) || hazard < 0.0
        throw(DomainError(hazard, "EscapeRate hazard must remain finite and non-negative"))
    end
    p_spike = -expm1(-rate * s.dt)
    if !isfinite(p_spike) || p_spike < 0.0 || p_spike > 1.0
        throw(DomainError(p_spike, "EscapeRate spike probability must remain finite and bounded"))
    end
    if rand() < p_spike
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
           isfinite(s.dt) && s.dt > 0.0
end

function safe_exp(x::Float64)
    return exp(clamp(x, -700.0, 700.0))
end

function reset!(s::EscapeRateNeuronState)::Nothing
    s.v = s.v_rest
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EscapeRateNeuronState()
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

end # module EscapeRateAccel
