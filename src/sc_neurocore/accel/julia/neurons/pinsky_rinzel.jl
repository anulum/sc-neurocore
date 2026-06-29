# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pinsky_rinzel (PR1994, RK4)

module PinskyRinzelAccel

export step!, simulate, PinskyRinzelNeuronState

# Pinsky-Rinzel 1994 two-compartment CA3 cell, fourth-order Runge-Kutta.
# Mirror of neurons/models/pinsky_rinzel.py: eight states
# (v_s, v_d, h, n, s, c, q, ca), chi(ca), capacitance cm. Kinetics: ModelDB 35358.

mutable struct PinskyRinzelNeuronState
    v_s::Float64
    v_d::Float64
    h::Float64
    n::Float64
    s::Float64
    c::Float64
    q::Float64
    ca::Float64
    cm::Float64
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
    PinskyRinzelNeuronState(-60.0, -60.0, 0.999, 0.001, 0.009, 0.007, 0.01, 0.2, 3.0,
        2.1, 0.5, 30.0, 15.0, 10.0, 0.8, 15.0, 0.1, 60.0, -75.0, 80.0, -60.0, 0.02, -20.0)
end

function _valid(s::PinskyRinzelNeuronState)
    values = (
        s.v_s, s.v_d, s.h, s.n, s.s, s.c, s.q, s.ca, s.cm, s.gc, s.p, s.g_na, s.g_kdr,
        s.g_ca, s.g_kahp, s.g_kc, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_l, s.dt, s.v_threshold,
    )
    return all(isfinite, values) &&
        0.0 <= s.h <= 1.0 &&
        0.0 <= s.n <= 1.0 &&
        0.0 <= s.s <= 1.0 &&
        0.0 <= s.c <= 1.0 &&
        0.0 <= s.q <= 1.0 &&
        s.ca >= 0.0 &&
        s.cm > 0.0 &&
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

# Traub activation rate a*dv / (1 - exp(-dv/k)), removable limit a*k.
function _exprel_minus(a::Float64, dv::Float64, k::Float64)
    abs(dv) < 1e-6 && return a * k
    return a * dv / (1.0 - exp(-dv / k))
end

# Traub deactivation rate a*dv / (exp(dv/k) - 1), removable limit a*k.
function _exprel_plus(a::Float64, dv::Float64, k::Float64)
    abs(dv) < 1e-6 && return a * k
    return a * dv / (exp(dv / k) - 1.0)
end

function _derivatives(s::PinskyRinzelNeuronState, y::NTuple{8,Float64}, i_s::Float64, i_d::Float64)
    v_s, v_d, h, n, sg, c, q, ca = y
    am = _exprel_minus(0.32, v_s + 46.9, 4.0)
    bm = _exprel_plus(0.28, v_s + 19.9, 5.0)
    m_inf = (am + bm) > 0.0 ? am / (am + bm) : 0.0
    ah = 0.128 * exp(-(v_s + 43.0) / 18.0)
    bh = 4.0 / (1.0 + exp(-(v_s + 20.0) / 5.0))
    an = _exprel_minus(0.016, v_s + 24.9, 5.0)
    bn = 0.25 * exp(-1.0 - 0.025 * v_s)
    a_s = 1.6 / (1.0 + exp(-0.072 * (v_d - 5.0)))
    b_s = _exprel_plus(0.02, v_d + 8.9, 5.0)
    if v_d <= -10.0
        ac = exp((v_d + 50.0) / 11.0 - (v_d + 53.5) / 27.0) / 18.975
        bc = 2.0 * exp((-53.5 - v_d) / 27.0) - ac
    else
        ac = 2.0 * exp((-53.5 - v_d) / 27.0)
        bc = 0.0
    end
    aq = min(0.00002 * ca, 0.01)
    bq = 0.001
    chi = min(ca / 250.0, 1.0)

    i_na = s.g_na * m_inf^2 * h * (v_s - s.e_na)
    i_kdr = s.g_kdr * n * (v_s - s.e_k)
    i_ls = s.g_l * (v_s - s.e_l)
    i_ca = s.g_ca * sg^2 * (v_d - s.e_ca)
    i_kahp = s.g_kahp * q * (v_d - s.e_k)
    i_kc = s.g_kc * c * chi * (v_d - s.e_k)
    i_ld = s.g_l * (v_d - s.e_l)
    i_coupling = s.gc * (v_d - v_s)

    dv_s = (-i_ls - i_na - i_kdr + i_coupling / s.p + i_s / s.p) / s.cm
    dv_d = (-i_ld - i_ca - i_kahp - i_kc - i_coupling / (1.0 - s.p) + i_d / (1.0 - s.p)) / s.cm
    return (
        dv_s, dv_d,
        ah * (1.0 - h) - bh * h,
        an * (1.0 - n) - bn * n,
        a_s * (1.0 - sg) - b_s * sg,
        ac * (1.0 - c) - bc * c,
        aq * (1.0 - q) - bq * q,
        -0.13 * i_ca - 0.075 * ca,
    )
end

_axpy(y::NTuple{8,Float64}, k::NTuple{8,Float64}, f::Float64) = ntuple(i -> y[i] + f * k[i], 8)

function step!(s::PinskyRinzelNeuronState, current_soma::Float64=0.0; current_dend::Float64=0.0, dt::Float64=s.dt)
    if !_valid(s) || !isfinite(current_soma) || !isfinite(current_dend) || !isfinite(dt) || dt <= 0.0
        return -1
    end
    s.dt = dt
    v_prev = s.v_s
    y = (s.v_s, s.v_d, s.h, s.n, s.s, s.c, s.q, s.ca)
    k1 = _derivatives(s, y, current_soma, current_dend)
    k2 = _derivatives(s, _axpy(y, k1, dt / 2.0), current_soma, current_dend)
    k3 = _derivatives(s, _axpy(y, k2, dt / 2.0), current_soma, current_dend)
    k4 = _derivatives(s, _axpy(y, k3, dt), current_soma, current_dend)
    nxt = ntuple(i -> y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]), 8)
    if !all(isfinite, nxt)
        return -1
    end
    s.v_s = nxt[1]
    s.v_d = nxt[2]
    s.h = clamp(nxt[3], 0.0, 1.0)
    s.n = clamp(nxt[4], 0.0, 1.0)
    s.s = clamp(nxt[5], 0.0, 1.0)
    s.c = clamp(nxt[6], 0.0, 1.0)
    s.q = clamp(nxt[7], 0.0, 1.0)
    s.ca = max(nxt[8], 0.0)
    return (s.v_s >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=0.75, dt::Float64=0.02)
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
