# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pinsky_rinzel

module PinskyRinzelAccel

export step!, simulate, PinskyRinzelNeuronState

mutable struct PinskyRinzelNeuronState
    v_s::Float64
    v_d::Float64
    h::Float64
    n::Float64
    s::Float64
    c::Float64
    q::Float64
    gc::Float64
    p::Float64
    g_na::Float64
    g_kdr::Float64
    g_ca::Float64
    g_kahp::Float64
    g_kc::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function PinskyRinzelNeuronState()
    PinskyRinzelNeuronState(-60.0, -60.0, 0.9, 0.1, 0.0, 0.0, 0.0, 2.1, 0.5, 30.0, 15.0, 10.0, 0.8, 15.0, 0.1, 60.0, -75.0, 80.0, -60.0, 0.02, -20.0)
end

function _valid(s::PinskyRinzelNeuronState)
    values = (
        s.v_s, s.v_d, s.h, s.n, s.s, s.c, s.q, s.gc, s.p, s.g_na, s.g_kdr,
        s.g_ca, s.g_kahp, s.g_kc, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_l, s.dt,
        s.v_threshold,
    )
    return all(isfinite, values) &&
        0.0 <= s.h <= 1.0 &&
        0.0 <= s.n <= 1.0 &&
        0.0 <= s.s <= 1.0 &&
        s.c >= 0.0 &&
        0.0 <= s.q <= 1.0 &&
        s.gc > 0.0 &&
        0.0 < s.p < 1.0 &&
        s.g_na > 0.0 &&
        s.g_kdr > 0.0 &&
        s.g_ca > 0.0 &&
        s.g_kahp > 0.0 &&
        s.g_kc > 0.0 &&
        s.g_l > 0.0 &&
        s.dt > 0.0
end

function _logistic(value::Float64)
    if value >= 0.0
        return 1.0 / (1.0 + exp(-value))
    end
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)
end

function _alpha(scale::Float64, x::Float64, divisor::Float64, fallback::Float64, positive_exp::Bool)
    abs(x) <= 1e-6 && return fallback
    if positive_exp
        return scale * x / (exp(x / divisor) - 1.0)
    end
    return scale * x / (1.0 - exp(-x / divisor))
end

function step!(s::PinskyRinzelNeuronState, current_soma::Float64=0.0; current_dend::Float64=0.0, dt::Float64=s.dt)
    if !_valid(s) || !isfinite(current_soma) || !isfinite(current_dend) || !isfinite(dt) || dt <= 0.0
        return -1
    end
    v_prev = s.v_s
    am = _alpha(0.32, s.v_s + 54.0, 4.0, 8.0, false)
    bm = _alpha(0.28, s.v_s + 27.0, 5.0, 5.6, true)
    m_inf = am / (am + bm)
    ah = 0.128 * exp(-(s.v_s + 50.0) / 18.0)
    bh = 4.0 * _logistic((s.v_s + 27.0) / 5.0)
    an = _alpha(0.032, s.v_s + 52.0, 5.0, 0.32, false)
    bn = 0.5 * exp(-(s.v_s + 57.0) / 40.0)
    s_inf = _logistic((s.v_d + 20.0) / 9.0)
    i_na = s.g_na * m_inf ^ 2 * s.h * (s.v_s - s.e_na)
    i_kdr = s.g_kdr * s.n * (s.v_s - s.e_k)
    i_ls = s.g_l * (s.v_s - s.e_l)
    i_ds = s.gc / s.p * (s.v_s - s.v_d)
    i_ca = s.g_ca * s.s ^ 2 * (s.v_d - s.e_ca)
    i_kahp = s.g_kahp * s.q * (s.v_d - s.e_k)
    chi = (s.v_d <= 50.0) ? min(s.v_d / 250.0 + 0.5, 1.0) : 2.0
    i_kc = s.g_kc * s.c * chi * (s.v_d - s.e_k)
    i_ld = s.g_l * (s.v_d - s.e_l)
    i_sd = s.gc / (1.0 - s.p) * (s.v_d - s.v_s)
    candidate = PinskyRinzelNeuronState(
        s.v_s + (-i_na - i_kdr - i_ls - i_ds + current_soma / s.p) * dt,
        s.v_d + (-i_ca - i_kahp - i_kc - i_ld - i_sd + current_dend / (1.0 - s.p)) * dt,
        s.h + (ah * (1.0 - s.h) - bh * s.h) * dt,
        s.n + (an * (1.0 - s.n) - bn * s.n) * dt,
        s.s + ((s_inf - s.s) / 5.0) * dt,
        max(0.0, s.c + (-0.13 * i_ca - 0.075 * s.c) * dt),
        s.q,
        s.gc, s.p, s.g_na, s.g_kdr, s.g_ca, s.g_kahp, s.g_kc, s.g_l,
        s.e_na, s.e_k, s.e_ca, s.e_l, dt, s.v_threshold,
    )
    q_inf = min(candidate.c / (candidate.c + 2.0), 1.0)
    candidate.q = s.q + ((q_inf - s.q) / 100.0) * dt
    if !_valid(candidate)
        return -1
    end
    s.v_s = candidate.v_s
    s.v_d = candidate.v_d
    s.h = candidate.h
    s.n = candidate.n
    s.s = candidate.s
    s.c = candidate.c
    s.q = candidate.q
    s.dt = candidate.dt
    return (s.v_s >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PinskyRinzelNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PinskyRinzelAccel
