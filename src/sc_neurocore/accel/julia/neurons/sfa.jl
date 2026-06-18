# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sfa

module SfaAccel

export step!, simulate, SFANeuronState, validate

const V_MIN = -200.0
const V_MAX = 100.0
const G_MAX = 1.0e9

mutable struct SFANeuronState
    v::Float64
    g_sfa::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_sfa::Float64
    delta_g::Float64
    e_k::Float64
    resistance::Float64
    dt::Float64
end

function SFANeuronState()
    SFANeuronState(-70.0, 0.0, -70.0, -70.0, -50.0, 10.0, 200.0, 0.5, -80.0, 1.0, 1.0)
end

_finite(value::Float64) = isfinite(value)
_nonnegative(value::Float64) = isfinite(value) && value >= 0.0

function validate(s::SFANeuronState)
    if any(x -> !_finite(x), (s.v, s.v_rest, s.v_reset, s.v_threshold, s.e_k))
        return false
    end
    if !(V_MIN <= s.v <= V_MAX) || !(V_MIN <= s.v_reset <= V_MAX)
        return false
    end
    if !_nonnegative(s.g_sfa) || s.g_sfa > G_MAX
        return false
    end
    if any(x -> !_finite(x) || x <= 0.0, (s.tau_m, s.tau_sfa, s.resistance, s.dt))
        return false
    end
    return _nonnegative(s.delta_g) && s.delta_g <= G_MAX
end

function _derivatives(s::SFANeuronState, v::Float64, g_sfa::Float64, I_ext::Float64)
    dv = (-(v - s.v_rest) - g_sfa * (v - s.e_k) + s.resistance * I_ext) / s.tau_m
    return dv, -g_sfa / s.tau_sfa
end

function _rk4_candidate(s::SFANeuronState, v::Float64, g_sfa::Float64, I_ext::Float64)
    k1v, k1g = _derivatives(s, v, g_sfa, I_ext)
    k2v, k2g = _derivatives(s, v + 0.5 * s.dt * k1v, g_sfa + 0.5 * s.dt * k1g, I_ext)
    k3v, k3g = _derivatives(s, v + 0.5 * s.dt * k2v, g_sfa + 0.5 * s.dt * k2g, I_ext)
    k4v, k4g = _derivatives(s, v + s.dt * k3v, g_sfa + s.dt * k3g, I_ext)
    return (
        v + (s.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        g_sfa + (s.dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g),
    )
end

function step!(s::SFANeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !_finite(dt) || dt <= 0.0
        return -1
    end
    original_dt = s.dt
    s.dt = dt
    if !_finite(I_ext) || !validate(s)
        s.dt = original_dt
        return -1
    end
    v_candidate, g_candidate = _rk4_candidate(s, s.v, s.g_sfa, I_ext)
    if any(x -> !isfinite(x), (v_candidate, g_candidate))
        s.dt = original_dt
        return -1
    end
    if !(V_MIN <= v_candidate <= V_MAX) || g_candidate < 0.0 || g_candidate > G_MAX
        s.dt = original_dt
        return -1
    end
    if v_candidate >= s.v_threshold
        g_after_spike = g_candidate + s.delta_g
        if !isfinite(g_after_spike) || g_after_spike > G_MAX
            s.dt = original_dt
            return -1
        end
        s.v = s.v_reset
        s.g_sfa = g_after_spike
        return 1
    end
    s.v = v_candidate
    s.g_sfa = g_candidate
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SFANeuronState()
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

end # module SfaAccel
