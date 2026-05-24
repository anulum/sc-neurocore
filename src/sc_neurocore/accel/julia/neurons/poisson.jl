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

function step!(s::PoissonNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate_poisson(s) || !isfinite(I_ext)
        return 0
    end

    rate_hz = I_ext < 0.0 ? s.rate_hz : I_ext
    if !isfinite(rate_hz) || rate_hz < 0.0
        return 0
    end
    p_spike = -expm1(-(rate_hz * s.dt_ms / 1000.0))
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
