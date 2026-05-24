# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for escape_rate

module EscapeRateAccel

export step!, simulate, validate_escape_rate, EscapeRateNeuronState

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

function step!(s::EscapeRateNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate_escape_rate(s) || !isfinite(I_ext)
        return 0
    end

    s.v += (-(s.v - s.v_rest) + s.resistance * I_ext) / s.tau_m * s.dt
    rate = s.rho_0 * safe_exp((s.v - s.v_threshold) / s.delta_u)
    p_spike = -expm1(-rate * s.dt)
    if rand() < p_spike
        s.v = s.v_reset
        return 1
    end
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
        return 0
    end
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
