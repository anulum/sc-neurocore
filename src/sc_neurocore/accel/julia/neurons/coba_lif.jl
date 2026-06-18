# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for coba_lif

module CobaLifAccel

export step!, simulate, COBALIFNeuronState, validate

const V_MIN = -200.0
const V_MAX = 100.0
const G_MAX = 1.0e9

mutable struct COBALIFNeuronState
    v::Float64
    g_e::Float64
    g_i::Float64
    c_m::Float64
    g_l::Float64
    e_l::Float64
    e_e::Float64
    e_i::Float64
    tau_e::Float64
    tau_i::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function COBALIFNeuronState()
    COBALIFNeuronState(-65.0, 0.0, 0.0, 200.0, 10.0, -65.0, 0.0, -80.0, 5.0, 10.0, -50.0, -65.0, 0.1)
end

_finite(value::Float64) = isfinite(value)
_nonnegative(value::Float64) = isfinite(value) && value >= 0.0

function validate(s::COBALIFNeuronState)
    if !_finite(s.v) || !(V_MIN <= s.v <= V_MAX)
        return false
    end
    if !_nonnegative(s.g_e) || !_nonnegative(s.g_i) || s.g_e > G_MAX || s.g_i > G_MAX
        return false
    end
    if any(x -> !_finite(x) || x <= 0.0, (s.c_m, s.tau_e, s.tau_i, s.dt))
        return false
    end
    if !_nonnegative(s.g_l)
        return false
    end
    if any(x -> !_finite(x), (s.e_l, s.e_e, s.e_i, s.v_threshold, s.v_reset))
        return false
    end
    return V_MIN <= s.v_reset <= V_MAX
end

function _derivatives(s::COBALIFNeuronState, v::Float64, g_e::Float64, g_i::Float64, I_ext::Float64)
    i_syn = g_e * (v - s.e_e) + g_i * (v - s.e_i)
    dv = (-s.g_l * (v - s.e_l) - i_syn + I_ext) / s.c_m
    return dv, -g_e / s.tau_e, -g_i / s.tau_i
end

function _rk4_candidate(s::COBALIFNeuronState, v::Float64, g_e::Float64, g_i::Float64, I_ext::Float64)
    k1v, k1e, k1i = _derivatives(s, v, g_e, g_i, I_ext)
    k2v, k2e, k2i = _derivatives(s, v + 0.5 * s.dt * k1v, g_e + 0.5 * s.dt * k1e, g_i + 0.5 * s.dt * k1i, I_ext)
    k3v, k3e, k3i = _derivatives(s, v + 0.5 * s.dt * k2v, g_e + 0.5 * s.dt * k2e, g_i + 0.5 * s.dt * k2i, I_ext)
    k4v, k4e, k4i = _derivatives(s, v + s.dt * k3v, g_e + s.dt * k3e, g_i + s.dt * k3i, I_ext)
    return (
        v + (s.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        g_e + (s.dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
        g_i + (s.dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
    )
end

function step!(s::COBALIFNeuronState, I_ext::Float64=0.0; delta_ge::Float64=0.0, delta_gi::Float64=0.0, dt::Float64=s.dt)
    if !_finite(dt) || dt <= 0.0
        return -1
    end
    original_dt = s.dt
    s.dt = dt
    if !_finite(I_ext) || !_nonnegative(delta_ge) || !_nonnegative(delta_gi) || !validate(s)
        s.dt = original_dt
        return -1
    end
    ge_pre = s.g_e + delta_ge
    gi_pre = s.g_i + delta_gi
    if ge_pre > G_MAX || gi_pre > G_MAX
        s.dt = original_dt
        return -1
    end
    i_syn = ge_pre * (s.v - s.e_e) + gi_pre * (s.v - s.e_i)
    v_candidate, ge_candidate, gi_candidate = _rk4_candidate(s, s.v, ge_pre, gi_pre, I_ext)
    if any(x -> !isfinite(x), (i_syn, v_candidate, ge_candidate, gi_candidate))
        s.dt = original_dt
        return -1
    end
    if !(V_MIN <= v_candidate <= V_MAX) || ge_candidate < 0.0 || gi_candidate < 0.0
        s.dt = original_dt
        return -1
    end
    if v_candidate >= s.v_threshold
        s.v = s.v_reset
        s.g_e = ge_candidate
        s.g_i = gi_candidate
        return 1
    end
    s.v = v_candidate
    s.g_e = ge_candidate
    s.g_i = gi_candidate
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = COBALIFNeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = result < 0 ? NaN : s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module CobaLifAccel
