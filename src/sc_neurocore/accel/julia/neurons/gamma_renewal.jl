# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gamma_renewal

module GammaRenewalAccel

export step!, simulate, validate_gamma_renewal, GammaRenewalNeuronState

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
    if !validate_gamma_renewal(s) || !isfinite(I_ext)
        return 0
    end
    r = I_ext < 0.0 ? s.rate_hz : I_ext
    if !isfinite(r) || r < 0.0
        return 0
    end
    s._time_since_spike += s.dt_ms / 1000.0
    p = spike_probability_at(s, s._time_since_spike, r)
    if rand() < p
        s._time_since_spike = 0.0
        return 1
    end
    return 0
end

function spike_probability_at(s::GammaRenewalNeuronState, elapsed_s::Float64, rate_hz::Float64)
    if !isfinite(elapsed_s) || elapsed_s < 0.0 || !isfinite(rate_hz) || rate_hz < 0.0
        return 0.0
    end
    if elapsed_s < 1e-12 || rate_hz == 0.0
        return 0.0
    end
    k = Int(s.shape_k)
    lam = k * rate_hz
    x = lam * elapsed_s
    log_f = k * log(lam) + (k - 1) * log(elapsed_s) - x - log_gamma_int(k)
    f = exp(clamp(log_f, -50.0, 50.0))
    survival = max(gamma_survival(k, x), 1e-15)
    hazard = f / survival
    return -expm1(-(hazard * s.dt_ms / 1000.0))
end

function validate_gamma_renewal(s::GammaRenewalNeuronState)
    return isfinite(s.rate_hz) && s.rate_hz >= 0.0 &&
           isfinite(s.shape_k) && s.shape_k == trunc(s.shape_k) && s.shape_k > 0.0 &&
           isfinite(s.dt_ms) && s.dt_ms > 0.0 &&
           isfinite(s._time_since_spike) && s._time_since_spike >= 0.0
end

function log_gamma_int(k::Int)
    return k > 1 ? sum(log(i) for i in 1:(k - 1)) : 0.0
end

function gamma_survival(k::Int, x::Float64)
    if k <= 0 || !isfinite(x)
        return 0.0
    end
    if x < 0.0
        return 1.0
    end
    s = 1.0
    term = 1.0
    for i in 1:(k - 1)
        term *= x / i
        s += term
    end
    return exp(-x) * s
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
