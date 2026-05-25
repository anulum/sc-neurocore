# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for poisson

module PoissonAccel

export step!, simulate, validate_poisson, PoissonNeuronState

mutable struct PoissonNeuronState
    rate_hz::Float64
    dt_ms::Float64
    _rng::Float64
end

function PoissonNeuronState()
    PoissonNeuronState(100.0, 1.0, 0.0)
end

function step!(s::PoissonNeuronState, I_ext::Float64=-1.0; dt::Float64=s.dt_ms)::Int
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "Poisson rate override must be finite"))
    end
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "Poisson dt_ms must be finite and positive"))
    end
    previous_dt = s.dt_ms
    s.dt_ms = dt
    if !validate_poisson(s)
        s.dt_ms = previous_dt
        throw(DomainError(s.rate_hz, "Poisson rate and timestep must be finite with non-negative rate and positive timestep"))
    end

    rate_hz = I_ext < 0.0 ? s.rate_hz : I_ext
    if !isfinite(rate_hz) || rate_hz < 0.0
        throw(DomainError(rate_hz, "Poisson active rate must be finite and non-negative"))
    end
    hazard = rate_hz * s.dt_ms / 1000.0
    if !isfinite(hazard) || hazard < 0.0
        throw(DomainError(hazard, "Poisson interval hazard must remain finite and non-negative"))
    end
    p_spike = -expm1(-hazard)
    if !isfinite(p_spike) || p_spike < 0.0 || p_spike > 1.0
        throw(DomainError(p_spike, "Poisson spike probability must remain finite and bounded"))
    end
    return rand() < p_spike ? 1 : 0
end

function validate_poisson(s::PoissonNeuronState)
    return isfinite(s.rate_hz) && s.rate_hz >= 0.0 && isfinite(s.dt_ms) && s.dt_ms > 0.0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PoissonNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.rate_hz
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PoissonAccel
