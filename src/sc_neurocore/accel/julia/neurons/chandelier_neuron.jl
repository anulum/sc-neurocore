# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for chandelier_neuron

module ChandelierNeuronAccel

export step!, simulate, validate, ChandelierNeuronState

mutable struct ChandelierNeuronState
    v::Float64
    h::Float64
    n::Float64
    d::Float64
    p::Float64
    g_na::Float64
    g_k::Float64
    g_kv1::Float64
    g_kv3::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function ChandelierNeuronState()
    ChandelierNeuronState(-65.0, 0.8, 0.1, 0.0, 0.0, 35.0, 9.0, 3.0, 4.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.01, -20.0)
end

_probability(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::ChandelierNeuronState)::Bool
    all(isfinite, (s.v, s.e_na, s.e_k, s.e_l, s.v_threshold)) &&
        all(_probability, (s.h, s.n, s.d, s.p)) &&
        all(x -> isfinite(x) && x >= 0.0, (s.g_na, s.g_k, s.g_kv1, s.g_kv3, s.g_l)) &&
        all(x -> isfinite(x) && x > 0.0, (s.c_m, s.phi, s.dt))
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

function step!(s::ChandelierNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end
    v_prev = s.v
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    v, h, n, d_gate, p_gate = s.v, s.h, s.n, s.d, s.p
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
        d_inf = 1.0 / (1.0 + _checked_exp(-(v + 50.0) / 10.0))
        d_next = d_gate + (d_inf - d_gate) / 150.0 * s.dt
        p_inf = 1.0 / (1.0 + _checked_exp(-(v + 10.0) / 10.0))
        p_next = p_gate + s.phi * (p_inf - p_gate) * s.dt
        i_na = s.g_na * m_inf ^ 3 * h_next * (v - s.e_na)
        i_k = s.g_k * n_next ^ 4 * (v - s.e_k)
        i_kv1 = s.g_kv1 * d_next ^ 4 * (v - s.e_k)
        i_kv3 = s.g_kv3 * p_next * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        v_next = v + (-i_na - i_k - i_kv1 - i_kv3 - i_l + I_ext) / s.c_m * s.dt
        if !(isfinite(v_next) && -100.0 <= v_next <= 60.0 && all(_probability, (h_next, n_next, d_next, p_next)))
            return -1
        end
        v, h, n, d_gate, p_gate = v_next, h_next, n_next, d_next, p_next
    end
    s.v, s.h, s.n, s.d, s.p = v, h, n, d_gate, p_gate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ChandelierNeuronState()
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

end # module ChandelierNeuronAccel
