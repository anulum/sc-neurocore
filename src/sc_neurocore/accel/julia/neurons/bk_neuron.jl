# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for bk_neuron

module BkNeuronAccel

export step!, simulate, validate, BKNeuronState

mutable struct BKNeuronState
    v::Float64
    h::Float64
    n::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_bk::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    tau_ca::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    sub_steps::Int
end

function BKNeuronState()
    BKNeuronState(-65.0, 0.6, 0.32, 0.0, 35.0, 9.0, 3.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 50.0, 0.5, -20.0, 1.0, 50)
end

_probability(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::BKNeuronState)::Bool
    all(isfinite, (s.v, s.e_na, s.e_k, s.e_l, s.v_threshold, s.gain)) &&
        _probability(s.h) && _probability(s.n) && isfinite(s.ca) && s.ca >= 0.0 &&
        all(x -> isfinite(x) && x >= 0.0, (s.g_na, s.g_k, s.g_bk, s.g_l)) &&
        all(x -> isfinite(x) && x > 0.0, (s.c_m, s.phi, s.tau_ca, s.dt)) &&
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

function step!(s::BKNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
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
    v, h, n, ca = s.v, s.h, s.n, s.ca
    fired = 0
    for _ in 1:s.sub_steps
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * _checked_exp(-(v + 44.0) / 80.0)
        ca_decay = max(ca + sub_dt * (-ca / s.tau_ca), 0.0)
        denom = ca_decay + 0.5
        if !isfinite(ca_decay) || !isfinite(denom) || denom <= 0.0
            return -1
        end
        v_half_bk = 10.0 - 30.0 * (ca_decay / denom)
        bk_inf = 1.0 / (1.0 + _checked_exp(-(v - v_half_bk) / 15.0))
        h_next = h + sub_dt * s.phi * (alpha_h * (1.0 - h) - beta_h * h)
        n_next = n + sub_dt * s.phi * (alpha_n * (1.0 - n) - beta_n * n)
        i_na = s.g_na * m_inf ^ 3 * h_next * (v - s.e_na)
        i_k = s.g_k * n_next ^ 4 * (v - s.e_k)
        i_bk = s.g_bk * bk_inf * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        dv = (-i_na - i_k - i_bk - i_l + inp) / s.c_m
        v_next = v + sub_dt * dv
        ca_next = ca_decay
        if v_next >= s.v_threshold
            fired = 1
            v_next = -65.0
            ca_next += 0.3
        end
        if !(isfinite(v_next) && -100.0 <= v_next <= 60.0 && _probability(h_next) && _probability(n_next) && isfinite(ca_next) && ca_next >= 0.0 && _probability(bk_inf))
            return -1
        end
        v, h, n, ca = v_next, h_next, n_next, ca_next
    end
    s.v, s.h, s.n, s.ca = v, h, n, ca
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BKNeuronState()
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

end # module BkNeuronAccel
