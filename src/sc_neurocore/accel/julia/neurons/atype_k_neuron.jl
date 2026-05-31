# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for atype_k_neuron

module AtypeKNeuronAccel

export step!, simulate, validate, ATypeKNeuronState

mutable struct ATypeKNeuronState
    v::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    g_na::Float64
    g_k::Float64
    g_a::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    sub_steps::Int
end

function ATypeKNeuronState()
    ATypeKNeuronState(-65.0, 0.6, 0.32, 0.1, 0.8, 35.0, 9.0, 8.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.5, -20.0, 1.0, 50)
end

_probability(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::ATypeKNeuronState)::Bool
    all(isfinite, (s.v, s.e_na, s.e_k, s.e_l, s.v_threshold, s.gain)) &&
        all(_probability, (s.h, s.n, s.a, s.b)) &&
        all(x -> isfinite(x) && x >= 0.0, (s.g_na, s.g_k, s.g_a, s.g_l)) &&
        all(x -> isfinite(x) && x > 0.0, (s.c_m, s.phi, s.dt)) &&
        s.sub_steps > 0
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

function step!(s::ATypeKNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end
    inp = s.gain * I_ext
    if !isfinite(inp)
        return -1
    end
    sub_dt = s.dt / s.sub_steps
    if !isfinite(sub_dt) || sub_dt <= 0.0
        return -1
    end
    v, h, n, a_gate, b_gate = s.v, s.h, s.n, s.a, s.b
    fired = 0
    for _ in 1:s.sub_steps
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * _checked_exp(-(v + 44.0) / 80.0)
        a_inf = 1.0 / (1.0 + _checked_exp(-(v + 50.0) / 20.0))
        b_inf = 1.0 / (1.0 + _checked_exp((v + 70.0) / 6.0))
        h_next = h + sub_dt * s.phi * (alpha_h * (1.0 - h) - beta_h * h)
        n_next = n + sub_dt * s.phi * (alpha_n * (1.0 - n) - beta_n * n)
        a_next = a_gate + sub_dt * (a_inf - a_gate) / 2.0
        b_next = b_gate + sub_dt * (b_inf - b_gate) / 50.0
        i_na = s.g_na * m_inf ^ 3 * h_next * (v - s.e_na)
        i_k = s.g_k * n_next ^ 4 * (v - s.e_k)
        i_a = s.g_a * a_next ^ 3 * b_next * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        dv = (-i_na - i_k - i_a - i_l + inp) / s.c_m
        v_next = v + sub_dt * dv
        if v_next >= s.v_threshold
            fired = 1
            v_next = -65.0
        end
        if !(isfinite(v_next) && -100.0 <= v_next <= 60.0 && _probability(h_next) && _probability(n_next) && _probability(a_next) && _probability(b_next))
            return -1
        end
        v, h, n, a_gate, b_gate = v_next, h_next, n_next, a_next, b_next
    end
    s.v, s.h, s.n, s.a, s.b = v, h, n, a_gate, b_gate
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ATypeKNeuronState()
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

end # module AtypeKNeuronAccel
