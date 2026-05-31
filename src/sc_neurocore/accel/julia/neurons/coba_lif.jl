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

function _decay(dt::Float64, tau::Float64)
    ratio = -dt / tau
    decay = ratio < -700.0 ? 0.0 : exp(ratio)
    if !_finite(decay) || decay < 0.0 || decay >= 1.0
        error("decay must be in [0, 1)")
    end
    return decay
end

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

function step!(s::COBALIFNeuronState, I_ext::Float64=0.0; delta_ge::Float64=0.0, delta_gi::Float64=0.0, dt::Float64=s.dt)
    if !_finite(I_ext) || !_nonnegative(delta_ge) || !_nonnegative(delta_gi) || !validate(s)
        return -1
    end
    decay_e = _decay(s.dt, s.tau_e)
    decay_i = _decay(s.dt, s.tau_i)
    ge_pre = s.g_e + delta_ge
    gi_pre = s.g_i + delta_gi
    if ge_pre > G_MAX || gi_pre > G_MAX
        return -1
    end
    i_syn = ge_pre * (s.v - s.e_e) + gi_pre * (s.v - s.e_i)
    dv = (-s.g_l * (s.v - s.e_l) - i_syn + I_ext) / s.c_m * s.dt
    v_candidate = s.v + dv
    ge_candidate = ge_pre * decay_e
    gi_candidate = gi_pre * decay_i
    if any(x -> !isfinite(x), (i_syn, dv, v_candidate, ge_candidate, gi_candidate))
        return -1
    end
    if !(V_MIN <= v_candidate <= V_MAX) || ge_candidate < 0.0 || gi_candidate < 0.0
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
