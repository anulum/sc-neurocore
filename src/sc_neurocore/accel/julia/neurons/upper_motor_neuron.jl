# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for upper_motor_neuron

module UpperMotorNeuronAccel

export step!, simulate, UpperMotorNeuronState

mutable struct UpperMotorNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    p::Float64
    s::Float64
    g_na::Float64
    g_k::Float64
    g_m::Float64
    g_ca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function UpperMotorNeuronState()
    UpperMotorNeuronState(-70.0, 0.05, 0.6, 0.3, 0.0, 0.0, 50.0, 5.0, 0.07, 0.3, 0.1, 50.0, -90.0, 120.0, -70.0, 1.0, 0.025, -20.0)
end

function _all_finite(values...)
    all(isfinite, values)
end

function _rate_exp(value::Float64)
    exp(max(-60.0, min(60.0, value)))
end

function _gate(previous::Float64, alpha::Float64, beta::Float64, dt::Float64)
    total = alpha + beta
    if total <= 0.0 || !isfinite(total)
        throw(ArgumentError("gate rates must define a positive finite time constant"))
    end
    steady = alpha / total
    max(0.0, min(1.0, steady + (previous - steady) * _rate_exp(-total * dt)))
end

function _gate_inf(previous::Float64, steady::Float64, tau::Float64, dt::Float64)
    if tau <= 0.0 || !isfinite(tau)
        throw(ArgumentError("gate time constant must be positive and finite"))
    end
    max(0.0, min(1.0, steady + (previous - steady) * _rate_exp(-dt / tau)))
end

function _validate_configuration(s::UpperMotorNeuronState, dt::Float64)
    if !_all_finite(s.g_na, s.g_k, s.g_m, s.g_ca, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_l, s.c_m, s.dt, dt, s.v_threshold)
        throw(ArgumentError("configuration values must be finite"))
    end
    if s.g_na < 0.0 || s.g_k < 0.0 || s.g_m < 0.0 || s.g_ca < 0.0 || s.g_l < 0.0
        throw(ArgumentError("conductances must be non-negative"))
    end
    if s.c_m <= 0.0
        throw(ArgumentError("c_m must be positive"))
    end
    if dt <= 0.0
        throw(ArgumentError("dt must be positive"))
    end
end

function _validate_state(s::UpperMotorNeuronState)
    if !_all_finite(s.v, s.m, s.h, s.n, s.p, s.s)
        throw(ArgumentError("state values must be finite"))
    end
    if s.v < -150.0 || s.v > 100.0
        throw(ArgumentError("v state must remain within [-150, 100] mV"))
    end
    if !(0.0 <= s.m <= 1.0) || !(0.0 <= s.h <= 1.0) || !(0.0 <= s.n <= 1.0) || !(0.0 <= s.p <= 1.0) || !(0.0 <= s.s <= 1.0)
        throw(ArgumentError("gate states must remain within [0, 1]"))
    end
end

function _step_candidate(s::UpperMotorNeuronState, v::Float64, m::Float64, h::Float64, n::Float64, p::Float64, ca_gate::Float64, I_ext::Float64, dt::Float64)
    dv = v - (-56.2)
    x_m = dv - 13.0
    alpha_m = (abs(x_m) < 1e-06) ? 0.32 * 4.0 : -0.32 * x_m / (_rate_exp(-x_m / 4.0) - 1.0)
    # Published Pospischil beta_m numerator is V - V_T - 40 (an earlier -17 offset,
    # shared with alpha_h, drove the cell into depolarisation block).
    x_bm = dv - 40.0
    beta_m = (abs(x_bm) < 1e-06) ? 0.28 * 5.0 : 0.28 * x_bm / (_rate_exp(x_bm / 5.0) - 1.0)
    alpha_h = 0.128 * _rate_exp(-(dv - 17.0) / 18.0)
    beta_h = 4.0 / (1.0 + _rate_exp(-(dv - 40.0) / 5.0))
    x_n = dv - 15.0
    alpha_n = (abs(x_n) < 1e-06) ? 0.032 * 5.0 : -0.032 * x_n / (_rate_exp(-x_n / 5.0) - 1.0)
    beta_n = 0.5 * _rate_exp(-(dv - 10.0) / 40.0)
    next_m = _gate(m, alpha_m, beta_m, dt)
    next_h = _gate(h, alpha_h, beta_h, dt)
    next_n = _gate(n, alpha_n, beta_n, dt)
    p_inf = 1.0 / (1.0 + _rate_exp(-(v + 35.0) / 10.0))
    tau_p = 400.0 / (3.3 * _rate_exp((v + 35.0) / 20.0) + _rate_exp(-(v + 35.0) / 20.0))
    next_p = _gate_inf(p, p_inf, tau_p, dt)
    s_inf = 1.0 / (1.0 + _rate_exp(-(v + 20.0) / 5.0))
    next_s = _gate_inf(ca_gate, s_inf, 10.0, dt)
    g_na = s.g_na * next_m ^ 3 * next_h
    g_k = s.g_k * next_n ^ 4
    g_m = s.g_m * next_p
    g_ca = s.g_ca * next_s ^ 2
    g_total = g_na + g_k + g_m + g_ca + s.g_l
    if g_total <= 0.0 || !isfinite(g_total)
        throw(ArgumentError("total conductance must be positive and finite"))
    end
    steady_v = (I_ext + g_na * s.e_na + g_k * s.e_k + g_m * s.e_k + g_ca * s.e_ca + s.g_l * s.e_l) / g_total
    next_v = steady_v + (v - steady_v) * _rate_exp(-(g_total / s.c_m) * dt)
    return max(-150.0, min(100.0, next_v)), next_m, next_h, next_n, next_p, next_s
end

function step!(s::UpperMotorNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    _validate_configuration(s, dt)
    _validate_state(s)
    if !isfinite(I_ext)
        throw(ArgumentError("current must be finite"))
    end
    v_prev = s.v
    v, m, h, n, p, ca_gate = s.v, s.m, s.h, s.n, s.p, s.s
    for _ in 1:4
        v, m, h, n, p, ca_gate = _step_candidate(s, v, m, h, n, p, ca_gate, I_ext, dt)
    end
    if !_all_finite(v, m, h, n, p, ca_gate)
        throw(ArgumentError("candidate state must be finite"))
    end
    s.v, s.m, s.h, s.n, s.p, s.s = v, m, h, n, p, ca_gate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.025)
    if n_steps < 0
        throw(ArgumentError("n_steps must be non-negative"))
    end
    s = UpperMotorNeuronState()
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

end # module UpperMotorNeuronAccel
