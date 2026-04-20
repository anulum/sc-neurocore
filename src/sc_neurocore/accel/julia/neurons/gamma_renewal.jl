# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gamma_renewal

module GammaRenewalAccel

export step!, simulate, GammaRenewalNeuronState

mutable struct GammaRenewalNeuronState
    rate_hz::Float64
    shape_k::Float64
    dt_ms::Float64
    _time_since_spike::Float64
    _rng::Float64
end

function GammaRenewalNeuronState()
    GammaRenewalNeuronState(50.0, 3.0, 1.0, 0.0, 0.0)
end

function step!(s::GammaRenewalNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        r = (rate_override < 0) ? s.rate_hz : rate_override
        s._time_since_spike += s.dt_ms / 1000.0
        t = s._time_since_spike
        k = s.shape_k
        lam = k * r
        if t < 1e-12
            return 0
        end
        log_f = k * log(lam) + (k - 1) * log(t) - lam * t - _log_gamma_int(k)
        f = exp(clamp(log_f, -50.0, 50.0))
        survival = _gamma_survival(k, lam * t)
        if survival < 1e-15
            survival = 1e-15
        end
        hazard = f / survival
        p = hazard * s.dt_ms / 1000.0
        if s._rng.random() < min(p, 1.0)
            s._time_since_spike = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GammaRenewalNeuronState()
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

end # module GammaRenewalAccel
