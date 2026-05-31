# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for alpha_motor_neuron

module AlphaMotorNeuronAccel

export step!, simulate, validate, AlphaMotorNeuronState

mutable struct AlphaMotorNeuronState
    v::Float64
    h::Float64
    n::Float64
    m_pic::Float64
    h_pic::Float64
    ca::Float64
    ca_buf::Float64
    g_na::Float64
    g_k::Float64
    g_pic::Float64
    g_ahp::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    tau_ca::Float64
    buf_ratio::Float64
    dt::Float64
    v_threshold::Float64
end

function AlphaMotorNeuronState()
    AlphaMotorNeuronState(-65.0, 0.8, 0.1, 0.0, 1.0, 0.0, 0.0, 35.0, 9.0, 0.15, 3.0, 0.3, 55.0, -90.0, 120.0, -65.0, 1.5, 4.0, 150.0, 0.003, 0.01, -20.0)
end

function _probability(x::Float64)::Bool
    isfinite(x) && 0.0 <= x <= 1.0
end

function validate(s::AlphaMotorNeuronState)::Bool
    all(isfinite, (s.v, s.e_na, s.e_k, s.e_ca, s.e_l, s.v_threshold)) &&
        _probability(s.h) && _probability(s.n) && _probability(s.m_pic) && _probability(s.h_pic) &&
        isfinite(s.ca) && s.ca >= 0.0 && isfinite(s.ca_buf) && s.ca_buf >= 0.0 &&
        all(x -> isfinite(x) && x >= 0.0, (s.g_na, s.g_k, s.g_pic, s.g_ahp, s.g_l)) &&
        all(x -> isfinite(x) && x > 0.0, (s.c_m, s.phi, s.tau_ca, s.dt)) &&
        isfinite(s.buf_ratio) && 0.0 <= s.buf_ratio <= 1.0
end

function _checked_exp(x::Float64)::Float64
    if !isfinite(x) || x > 709.0
        return NaN
    elseif x < -745.0
        return 0.0
    end
    exp(x)
end

function _safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)::Float64
    d = v + vhalf
    if abs(d) < 1e-7
        return fallback
    end
    rate = a * d / (1.0 - _checked_exp(-d / k))
    isfinite(rate) ? rate : NaN
end

function step!(s::AlphaMotorNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end
    v_prev = s.v
    v = s.v
    h = s.h
    n = s.n
    m_pic = s.m_pic
    h_pic = s.h_pic
    ca = s.ca
    ca_buf = s.ca_buf
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    for _ in 1:n_sub
        am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        bm = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
        m_inf = am / (am + bm)
        ah = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        bn = 0.125 * _checked_exp(-(v + 44.0) / 80.0)
        h_next = h + s.phi * (ah * (1.0 - h) - bh * h) * s.dt
        n_next = n + s.phi * (an * (1.0 - n) - bn * n) * s.dt
        m_pic_inf = 1.0 / (1.0 + _checked_exp(-(v + 40.0) / 5.0))
        m_pic_next = m_pic + (m_pic_inf - m_pic) / 50.0 * s.dt
        h_pic_inf = 1.0 / (1.0 + _checked_exp((v + 40.0) / 8.0))
        tau_h_pic = 200.0 + 100.0 / max(0.01, 1.0 + ((v + 40.0) / 10.0) ^ 2)
        h_pic_next = clamp(h_pic + (h_pic_inf - h_pic) / tau_h_pic * s.dt, 0.0, 1.0)
        i_ca_entry = s.g_pic * m_pic_next * h_pic_next * (v - s.e_ca)
        ca_influx = (i_ca_entry < 0.0) ? -i_ca_entry * 0.001 : 0.0
        ca_spike = (v > -10.0) ? 0.02 : 0.0
        ca_next = max(0.0, ca + (-ca / s.tau_ca + (ca_influx + ca_spike) * s.buf_ratio) * s.dt)
        ca_buf_next = max(0.0, ca_buf + ((ca_influx + ca_spike) * (1.0 - s.buf_ratio) - ca_buf / (s.tau_ca * 5.0)) * s.dt)
        ca_total = ca_next + ca_buf_next * 0.01
        ahp_inf = ca_total ^ 2 / (ca_total ^ 2 + 0.25)
        i_na = s.g_na * m_inf ^ 3 * h_next * (v - s.e_na)
        i_k = s.g_k * n_next ^ 4 * (v - s.e_k)
        i_pic = s.g_pic * m_pic_next * h_pic_next * (v - s.e_ca)
        i_ahp = s.g_ahp * ahp_inf * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        v_next = v + (-i_na - i_k - i_pic - i_ahp - i_l + I_ext) / s.c_m * s.dt
        if !(isfinite(v_next) && _probability(h_next) && _probability(n_next) && _probability(m_pic_next) && _probability(h_pic_next) && isfinite(ca_next) && ca_next >= 0.0 && isfinite(ca_buf_next) && ca_buf_next >= 0.0)
            return -1
        end
        v, h, n, m_pic, h_pic, ca, ca_buf = v_next, h_next, n_next, m_pic_next, h_pic_next, ca_next, ca_buf_next
    end
    s.v = v
    s.h = h
    s.n = n
    s.m_pic = m_pic
    s.h_pic = h_pic
    s.ca = ca
    s.ca_buf = ca_buf
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AlphaMotorNeuronState()
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

end # module AlphaMotorNeuronAccel
