# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for dcn_neuron

module DcnNeuronAccel

export step!, simulate, DCNNeuronState

mutable struct DCNNeuronState
    v::Float64
    h::Float64
    n::Float64
    p::Float64
    s::Float64
    r::Float64
    ca::Float64
    g_na::Float64
    g_nap::Float64
    g_k::Float64
    g_t::Float64
    g_ahp::Float64
    g_h::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_h::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    tau_ca::Float64
    kd_ahp::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    _sub_steps::Float64
end

function DCNNeuronState()
    DCNNeuronState(-60.0, 0.6, 0.32, 0.01, 0.8, 0.1, 0.05, 35.0, 0.5, 9.0, 0.1, 2.0, 0.02, 0.2, 55.0, -90.0, 120.0, -40.0, -65.0, 1.0, 5.0, 150.0, 0.5, 0.5, -20.0, 1.0, 20.0)
end

function _safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    abs(d) < 1e-7 && return fallback
    return a * d / (1.0 - exp(-d / k))
end

function _clamp01(x::Float64)
    return max(0.0, min(1.0, x))
end

function _exact_relax(value::Float64, target::Float64, tau::Float64, dt::Float64)
    return target + (value - target) * exp(-dt / tau)
end

function _exact_hh_gate(value::Float64, alpha::Float64, beta::Float64, phi::Float64, dt::Float64)
    rate = phi * (alpha + beta)
    target = alpha / (alpha + beta)
    return target + (value - target) * exp(-rate * dt)
end

function _exact_voltage_step(v::Float64, input_current::Float64, c_m::Float64, dt::Float64, conductances)
    g_total = sum(pair[1] for pair in conductances)
    if g_total <= 0.0
        return v + dt * input_current / c_m
    end
    reversal_drive = sum(pair[1] * pair[2] for pair in conductances)
    v_inf = (input_current + reversal_drive) / g_total
    return v_inf + (v - v_inf) * exp(-dt * g_total / c_m)
end

function validate!(s::DCNNeuronState)
    values = (s.v, s.h, s.n, s.p, s.s, s.r, s.ca, s.g_na, s.g_nap, s.g_k, s.g_t,
        s.g_ahp, s.g_h, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_h, s.e_l, s.c_m,
        s.phi, s.tau_ca, s.kd_ahp, s.dt, s.v_threshold, s.gain, s._sub_steps)
    all(isfinite, values) || throw(ArgumentError("DCN state and parameters must be finite"))
    -100.0 <= s.v <= 60.0 || throw(ArgumentError("v must be in the physical clamp interval [-100, 60]"))
    all(0.0 .<= (s.h, s.n, s.p, s.s, s.r) .<= 1.0) || throw(ArgumentError("DCN gates must be in [0, 1]"))
    s.ca >= 0.0 || throw(ArgumentError("ca must be non-negative"))
    all((s.g_na, s.g_nap, s.g_k, s.g_t, s.g_ahp, s.g_h, s.g_l) .>= 0.0) || throw(ArgumentError("conductances must be non-negative"))
    s.c_m > 0.0 || throw(ArgumentError("c_m must be positive"))
    s.phi > 0.0 || throw(ArgumentError("phi must be positive"))
    s.tau_ca > 0.0 || throw(ArgumentError("tau_ca must be positive"))
    s.kd_ahp > 0.0 || throw(ArgumentError("kd_ahp must be positive"))
    s.dt > 0.0 || throw(ArgumentError("dt must be positive"))
    s.gain >= 0.0 || throw(ArgumentError("gain must be non-negative"))
    s._sub_steps >= 1.0 || throw(ArgumentError("_sub_steps must be positive"))
    return nothing
end

function step!(s::DCNNeuronState, I_ext::Float64=0.0)
    validate!(s)
    isfinite(I_ext) || throw(ArgumentError("current must be finite"))
    inp = s.gain * I_ext
    sub_dt = s.dt / s._sub_steps
    fired = 0
    v, h, n, p, q, r, ca = s.v, s.h, s.n, s.p, s.s, s.r, s.ca
    for _ in 1:Int(s._sub_steps)
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
        p_inf = 1.0 / (1.0 + exp(-(v + 48.0) / 5.0))
        tau_p = 5.0 + 15.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ^ 2)
        m_t_inf = 1.0 / (1.0 + exp(-(v + 52.0) / 5.0))
        s_inf = 1.0 / (1.0 + exp((v + 60.0) / 6.5))
        tau_s = 20.0 + 50.0 / (1.0 + exp((v + 65.0) / 10.0))
        r_inf = 1.0 / (1.0 + exp((v + 80.0) / 10.0))
        tau_r = 100.0 + 200.0 / (1.0 + exp((v + 70.0) / 10.0))
        h = _exact_hh_gate(h, alpha_h, beta_h, s.phi, sub_dt)
        n = _exact_hh_gate(n, alpha_n, beta_n, s.phi, sub_dt)
        p = _exact_relax(p, p_inf, tau_p, sub_dt)
        q = _exact_relax(q, s_inf, tau_s, sub_dt)
        r = _exact_relax(r, r_inf, tau_r, sub_dt)
        i_t = s.g_t * m_t_inf ^ 2 * q * (v - s.e_ca)
        ca_entry = (i_t < 0.0) ? -i_t * 0.001 : 0.0
        ca = _exact_relax(ca, ca_entry * s.tau_ca, s.tau_ca, sub_dt)
        ca = max(0.0, ca)
        ahp_inf = ca ^ 2 / (ca ^ 2 + s.kd_ahp ^ 2)
        g_na_eff = s.g_na * m_inf ^ 3 * h
        g_nap_eff = s.g_nap * p
        g_k_eff = s.g_k * n ^ 4
        g_t_eff = s.g_t * m_t_inf ^ 2 * q
        g_ahp_eff = s.g_ahp * ahp_inf
        g_h_eff = s.g_h * r
        v = _exact_voltage_step(v, inp, s.c_m, sub_dt, (
            (g_na_eff, s.e_na),
            (g_nap_eff, s.e_na),
            (g_k_eff, s.e_k),
            (g_t_eff, s.e_ca),
            (g_ahp_eff, s.e_k),
            (g_h_eff, s.e_h),
            (s.g_l, s.e_l),
        ))
        if v >= s.v_threshold
            fired = 1
            v = -60.0
            q *= 0.5
            ca += 0.5
        end
    end
    all(isfinite, (v, h, n, p, q, r, ca)) || throw(ArgumentError("DCN candidate state must be finite"))
    s.v = max(-100.0, min(60.0, v))
    s.h = _clamp01(h)
    s.n = _clamp01(n)
    s.p = _clamp01(p)
    s.s = _clamp01(q)
    s.r = _clamp01(r)
    s.ca = max(0.0, ca)
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = DCNNeuronState()
    s.dt = dt
    validate!(s)
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DcnNeuronAccel
