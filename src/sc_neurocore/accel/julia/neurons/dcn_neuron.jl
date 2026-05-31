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

function validate!(s::DCNNeuronState)
    values = (s.v, s.h, s.n, s.p, s.s, s.r, s.ca, s.g_na, s.g_nap, s.g_k, s.g_t,
        s.g_ahp, s.g_h, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_h, s.e_l, s.c_m,
        s.phi, s.tau_ca, s.kd_ahp, s.dt, s.v_threshold, s.gain, s._sub_steps)
    all(isfinite, values) || throw(ArgumentError("DCN state and parameters must be finite"))
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
        h += sub_dt * s.phi * (alpha_h * (1.0 - h) - beta_h * h)
        n += sub_dt * s.phi * (alpha_n * (1.0 - n) - beta_n * n)
        p += sub_dt * (p_inf - p) / tau_p
        q += sub_dt * (s_inf - q) / tau_s
        r += sub_dt * (r_inf - r) / tau_r
        i_t = s.g_t * m_t_inf ^ 2 * q * (v - s.e_ca)
        ca_entry = (i_t < 0.0) ? -i_t * 0.001 : 0.0
        ca += sub_dt * (ca_entry - ca / s.tau_ca)
        ca = max(0.0, ca)
        ahp_inf = ca ^ 2 / (ca ^ 2 + s.kd_ahp ^ 2)
        i_na = s.g_na * m_inf ^ 3 * h * (v - s.e_na)
        i_nap = s.g_nap * p * (v - s.e_na)
        i_k = s.g_k * n ^ 4 * (v - s.e_k)
        i_ahp = s.g_ahp * ahp_inf * (v - s.e_k)
        i_h = s.g_h * r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        dv_val = (-i_na - i_nap - i_k - i_t - i_ahp - i_h - i_l + inp) / s.c_m
        v += sub_dt * dv_val
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
