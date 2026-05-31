# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for cerebellar_basket_neuron

module CerebellarBasketNeuronAccel

export step!, simulate, validate, CerebellarBasketNeuronState

mutable struct CerebellarBasketNeuronState
    v::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_a::Float64
    g_kca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function CerebellarBasketNeuronState()
    CerebellarBasketNeuronState(-65.0, 0.8, 0.1, 0.0, 0.9, 0.05, 35.0, 9.0, 3.0, 2.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.01, -20.0)
end

_probability(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::CerebellarBasketNeuronState)::Bool
    all(isfinite, (s.v, s.e_na, s.e_k, s.e_l, s.v_threshold)) &&
        all(_probability, (s.h, s.n, s.a, s.b)) &&
        isfinite(s.ca) && s.ca >= 0.0 &&
        all(x -> isfinite(x) && x >= 0.0, (s.g_na, s.g_k, s.g_a, s.g_kca, s.g_l)) &&
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

function step!(s::CerebellarBasketNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end
    v_prev = s.v
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    v, h, n, a_gate, b_gate, ca = s.v, s.h, s.n, s.a, s.b, s.ca
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
        a_inf = 1.0 / (1.0 + _checked_exp(-(v + 45.0) / 15.0))
        b_inf = 1.0 / (1.0 + _checked_exp((v + 75.0) / 8.0))
        a_next = a_gate + s.phi * (a_inf - a_gate) / 5.0 * s.dt
        b_next = b_gate + (b_inf - b_gate) / 50.0 * s.dt
        denom = ca + 0.2
        if !isfinite(denom) || denom <= 0.0
            return -1
        end
        q_inf = ca / denom
        i_ca_entry = (v > -20.0) ? 0.01 * (v + 20.0) : 0.0
        ca_next = max(0.0, ca + (-ca / 80.0 + i_ca_entry) * s.dt)
        i_na = s.g_na * m_inf ^ 3 * h_next * (v - s.e_na)
        i_k = s.g_k * n_next ^ 4 * (v - s.e_k)
        i_a = s.g_a * a_next ^ 3 * b_next * (v - s.e_k)
        i_kca = s.g_kca * q_inf * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        v_next = v + (-i_na - i_k - i_a - i_kca - i_l + I_ext) / s.c_m * s.dt
        if !(isfinite(v_next) && -100.0 <= v_next <= 60.0 && all(_probability, (h_next, n_next, a_next, b_next)) && isfinite(ca_next) && ca_next >= 0.0 && _probability(q_inf))
            return -1
        end
        v, h, n, a_gate, b_gate, ca = v_next, h_next, n_next, a_next, b_next, ca_next
    end
    s.v, s.h, s.n, s.a, s.b, s.ca = v, h, n, a_gate, b_gate, ca
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CerebellarBasketNeuronState()
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

end # module CerebellarBasketNeuronAccel
