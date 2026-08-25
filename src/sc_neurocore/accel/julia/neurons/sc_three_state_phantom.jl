# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for retained three-state project phantom

module SCThreeStatePhantomAccel

export step!, simulate, SCThreeStatePhantomState, validate_state

const VOLTAGE_MIN = -250.0
const VOLTAGE_MAX = 250.0
const GATE_TOL = 1e-9

mutable struct SCThreeStatePhantomState
    v::Float64
    s1::Float64
    s2::Float64
    g_ca::Float64
    g_k::Float64
    g_s1::Float64
    g_s2::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    v_m::Float64
    s_m::Float64
    v_n::Float64
    s_n::Float64
    v_s1::Float64
    s_s1::Float64
    v_s2::Float64
    s_s2::Float64
    tau_s1::Float64
    tau_s2::Float64
    dt::Float64
    v_threshold::Float64
end

function SCThreeStatePhantomState()
    SCThreeStatePhantomState(-50.0, 0.1, 0.1, 3.6, 10.0, 4.0, 4.0, 0.2, 25.0, -75.0, -40.0, 5.3, -20.0, 12.0, -16.0, 5.6, -40.0, 10.0, -42.0, 0.4, 20000.0, 100000.0, 0.5, -20.0)
end

_positive(x) = isfinite(x) && x > 0.0
_nonnegative(x) = isfinite(x) && x >= 0.0
_gate(x) = isfinite(x) && 0.0 <= x <= 1.0

function validate_state(s::SCThreeStatePhantomState)
    return isfinite(s.v) && VOLTAGE_MIN <= s.v <= VOLTAGE_MAX && _gate(s.s1) && _gate(s.s2) &&
        _nonnegative(s.g_ca) && _nonnegative(s.g_k) && _nonnegative(s.g_s1) && _nonnegative(s.g_s2) && _nonnegative(s.g_l) &&
        isfinite(s.e_ca) && isfinite(s.e_k) && isfinite(s.e_l) && _positive(s.c_m) &&
        isfinite(s.v_m) && _positive(s.s_m) && isfinite(s.v_n) && _positive(s.s_n) &&
        isfinite(s.v_s1) && _positive(s.s_s1) && isfinite(s.v_s2) && _positive(s.s_s2) &&
        _positive(s.tau_s1) && _positive(s.tau_s2) && _positive(s.dt) && isfinite(s.v_threshold)
end

function _boltz(v::Float64, vh::Float64, k::Float64)
    z = (vh - v) / k
    if z >= 0.0
        exp_neg = exp(-z)
        return exp_neg / (1.0 + exp_neg)
    end
    exp_pos = exp(z)
    return 1.0 / (1.0 + exp_pos)
end

function _derivatives(s::SCThreeStatePhantomState, v::Float64, s1::Float64, s2::Float64, I_ext::Float64)
    m_inf = _boltz(v, s.v_m, s.s_m)
    n_inf = _boltz(v, s.v_n, s.s_n)
    s1_inf = _boltz(v, s.v_s1, s.s_s1)
    s2_inf = _boltz(v, s.v_s2, s.s_s2)
    i_ca = s.g_ca * m_inf * (v - s.e_ca)
    i_k = s.g_k * n_inf * (v - s.e_k)
    i_s1 = s.g_s1 * s1 * (v - s.e_k)
    i_s2 = s.g_s2 * s2 * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + I_ext) / s.c_m
    ds1 = (s1_inf - s1) / s.tau_s1
    ds2 = (s2_inf - s2) / s.tau_s2
    return dv, ds1, ds2
end

function _candidate_valid(v::Float64, s1::Float64, s2::Float64)
    return isfinite(v) && VOLTAGE_MIN <= v <= VOLTAGE_MAX &&
        isfinite(s1) && -GATE_TOL <= s1 <= 1.0 + GATE_TOL &&
        isfinite(s2) && -GATE_TOL <= s2 <= 1.0 + GATE_TOL
end

function step!(s::SCThreeStatePhantomState, I_ext::Float64=0.0; dt::Union{Nothing, Float64}=nothing)
    old_dt = s.dt
    if dt !== nothing
        s.dt = dt
    end
    try
        if !isfinite(I_ext) || !validate_state(s)
            return 0
        end
        v_prev = s.v
        h = s.dt
        k1 = _derivatives(s, s.v, s.s1, s.s2, I_ext)
        k2 = _derivatives(s, s.v + 0.5h * k1[1], s.s1 + 0.5h * k1[2], s.s2 + 0.5h * k1[3], I_ext)
        k3 = _derivatives(s, s.v + 0.5h * k2[1], s.s1 + 0.5h * k2[2], s.s2 + 0.5h * k2[3], I_ext)
        k4 = _derivatives(s, s.v + h * k3[1], s.s1 + h * k3[2], s.s2 + h * k3[3], I_ext)
        v = s.v + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        s1 = s.s1 + h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        s2 = s.s2 + h * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if !_candidate_valid(v, s1, s2)
            return 0
        end
        s.v = v
        s.s1 = clamp(s1, 0.0, 1.0)
        s.s2 = clamp(s2, 0.0, 1.0)
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    finally
        s.dt = old_dt
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    s = SCThreeStatePhantomState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SCThreeStatePhantomAccel
