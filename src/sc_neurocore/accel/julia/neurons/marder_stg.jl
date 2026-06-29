# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for marder_stg (LGMA98 STG, RK4)

module MarderStgAccel

export step!, simulate, MarderSTGNeuronState

# Liu-Golowasch-Marder-Abbott 1998 STG neuron, RK4. Mirror of
# neurons/models/marder_stg.py: thirteen states, voltage-dependent time
# constants, Nernst calcium reversal. ModelDB 93321.

mutable struct MarderSTGNeuronState
    v::Float64
    m_na::Float64
    h_na::Float64
    m_cat::Float64
    h_cat::Float64
    m_cas::Float64
    h_cas::Float64
    m_a::Float64
    h_a::Float64
    m_kca::Float64
    m_kd::Float64
    m_h::Float64
    ca::Float64
    cm::Float64
    g_na::Float64
    g_cat::Float64
    g_cas::Float64
    g_a::Float64
    g_kca::Float64
    g_kd::Float64
    g_h::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_h::Float64
    e_l::Float64
    ca_out::Float64
    ca_rest::Float64
    tau_ca::Float64
    f_ca::Float64
    celsius::Float64
    dt::Float64
    v_threshold::Float64
end

function MarderSTGNeuronState()
    MarderSTGNeuronState(-60.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05,
        1.0, 200.0, 2.5, 4.0, 50.0, 25.0, 75.0, 0.01, 0.01,
        50.0, -80.0, -20.0, -50.0, 3000.0, 0.05, 20.0, 0.94, 10.0, 0.05, -20.0)
end

_ms_exp(x::Float64) = exp(clamp(x, -700.0, 700.0))
_ms_sig(v::Float64, vh::Float64, s::Float64) = 1.0 / (1.0 + _ms_exp((vh - v) / s))

function _nernst_e_ca(s::MarderSTGNeuronState, ca::Float64)
    rt_zf = 1000.0 * 8.314462618 * (s.celsius + 273.15) / (2.0 * 96485.33212)
    return rt_zf * log(s.ca_out / max(ca, 1e-9))
end

function _derivatives(s::MarderSTGNeuronState, y::NTuple{13,Float64}, current::Float64)
    v, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, ca = y
    tau_m_na = 1.32 - 1.26 / (1.0 + _ms_exp(-(v + 120.0) / 25.0))
    tau_h_na = (0.67 / (1.0 + _ms_exp(-(v + 62.9) / 10.0))) * (1.5 + 1.0 / (1.0 + _ms_exp((v + 34.9) / 3.6)))
    tau_m_cat = 21.7 - 21.3 / (1.0 + _ms_exp(-(v + 68.1) / 20.5))
    tau_h_cat = 105.0 - 89.8 / (1.0 + _ms_exp(-(v + 55.0) / 16.9))
    tau_m_cas = 1.4 + 7.0 / (_ms_exp((v + 27.0) / 10.0) + _ms_exp(-(v + 70.0) / 13.0))
    tau_h_cas = 60.0 + 150.0 / (_ms_exp((v + 55.0) / 9.0) + _ms_exp(-(v + 65.0) / 16.0))
    tau_m_a = 11.6 - 10.4 / (1.0 + _ms_exp(-(v + 32.9) / 15.2))
    tau_h_a = 38.6 - 29.2 / (1.0 + _ms_exp(-(v + 38.9) / 26.5))
    tau_m_kca = 90.3 - 75.1 / (1.0 + _ms_exp(-(v + 46.0) / 22.7))
    tau_m_kd = 7.2 - 6.4 / (1.0 + _ms_exp(-(v + 28.3) / 19.2))
    tau_m_h = 272.0 + 1499.0 / (1.0 + _ms_exp(-(v + 42.2) / 8.73))

    m_kca_inf = (ca / (ca + 3.0)) * _ms_sig(v, -28.3, 12.6)
    e_ca = _nernst_e_ca(s, ca)
    i_na = s.g_na * m_na^3 * h_na * (v - s.e_na)
    i_cat = s.g_cat * m_cat^3 * h_cat * (v - e_ca)
    i_cas = s.g_cas * m_cas^3 * h_cas * (v - e_ca)
    i_a = s.g_a * m_a^3 * h_a * (v - s.e_k)
    i_kca = s.g_kca * m_kca^4 * (v - s.e_k)
    i_kd = s.g_kd * m_kd^4 * (v - s.e_k)
    i_h = s.g_h * m_h * (v - s.e_h)
    i_l = s.g_l * (v - s.e_l)

    dv = (current - i_na - i_cat - i_cas - i_a - i_kca - i_kd - i_h - i_l) / s.cm
    dca = (-s.f_ca * (i_cat + i_cas) - (ca - s.ca_rest)) / s.tau_ca
    return (
        dv,
        (_ms_sig(v, -25.5, 5.29) - m_na) / tau_m_na,
        (_ms_sig(v, -48.9, -5.18) - h_na) / tau_h_na,
        (_ms_sig(v, -27.1, 7.2) - m_cat) / tau_m_cat,
        (_ms_sig(v, -32.1, -5.5) - h_cat) / tau_h_cat,
        (_ms_sig(v, -33.0, 8.1) - m_cas) / tau_m_cas,
        (_ms_sig(v, -60.0, -6.2) - h_cas) / tau_h_cas,
        (_ms_sig(v, -27.2, 8.7) - m_a) / tau_m_a,
        (_ms_sig(v, -56.9, -4.9) - h_a) / tau_h_a,
        (m_kca_inf - m_kca) / tau_m_kca,
        (_ms_sig(v, -12.3, 11.8) - m_kd) / tau_m_kd,
        (_ms_sig(v, -70.0, -6.0) - m_h) / tau_m_h,
        dca,
    )
end

_axpy(y::NTuple{13,Float64}, k::NTuple{13,Float64}, f::Float64) = ntuple(i -> y[i] + f * k[i], 13)

function step!(s::MarderSTGNeuronState, current::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(current) || !isfinite(dt) || dt <= 0.0
        return -1
    end
    s.dt = dt
    v_prev = s.v
    y = (s.v, s.m_na, s.h_na, s.m_cat, s.h_cat, s.m_cas, s.h_cas, s.m_a, s.h_a, s.m_kca, s.m_kd, s.m_h, s.ca)
    k1 = _derivatives(s, y, current)
    k2 = _derivatives(s, _axpy(y, k1, dt / 2.0), current)
    k3 = _derivatives(s, _axpy(y, k2, dt / 2.0), current)
    k4 = _derivatives(s, _axpy(y, k3, dt), current)
    nxt = ntuple(i -> y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]), 13)
    if !all(isfinite, nxt)
        return -1
    end
    s.v = nxt[1]
    s.m_na = clamp(nxt[2], 0.0, 1.0)
    s.h_na = clamp(nxt[3], 0.0, 1.0)
    s.m_cat = clamp(nxt[4], 0.0, 1.0)
    s.h_cat = clamp(nxt[5], 0.0, 1.0)
    s.m_cas = clamp(nxt[6], 0.0, 1.0)
    s.h_cas = clamp(nxt[7], 0.0, 1.0)
    s.m_a = clamp(nxt[8], 0.0, 1.0)
    s.h_a = clamp(nxt[9], 0.0, 1.0)
    s.m_kca = clamp(nxt[10], 0.0, 1.0)
    s.m_kd = clamp(nxt[11], 0.0, 1.0)
    s.m_h = clamp(nxt[12], 0.0, 1.0)
    s.ca = max(nxt[13], 0.0)
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=100000; I_ext::Float64=0.0, dt::Float64=0.05)
    s = MarderSTGNeuronState()
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

end # module MarderStgAccel
