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
    DCNNeuronState(-60.0, 0.6, 0.32, 0.01, 0.8, 0.1, 0.05, 35.0, 0.5, 9.0, 0.1, 2.0, 0.02, 0.2, 55.0, -90.0, 120.0, -40.0, -65.0, 1.0, 5.0, 150.0, 0.5, 0.5, -20.0, 1.0, 0.0)
end

function step!(s::DCNNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    inp = s.gain * I_ext
    sub_dt = s.dt / s._sub_steps
    fired = 0
    for _ in 1:s._sub_steps
        v = s.v
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
        s.h += sub_dt * s.phi * (alpha_h * (1.0 - s.h) - beta_h * s.h)
        s.n += sub_dt * s.phi * (alpha_n * (1.0 - s.n) - beta_n * s.n)
        s.p += sub_dt * (p_inf - s.p) / tau_p
        s.s += sub_dt * (s_inf - s.s) / tau_s
        s.r += sub_dt * (r_inf - s.r) / tau_r
        i_t = s.g_t * m_t_inf ^ 2 * s.s * (v - s.e_ca)
        ca_entry = (i_t < 0.0) ? -i_t * 0.001 : 0.0
        s.ca += sub_dt * (ca_entry - s.ca / s.tau_ca)
        s.ca = max(0.0, s.ca)
        ahp_inf = s.ca ^ 2 / (s.ca ^ 2 + s.kd_ahp ^ 2)
        i_na = s.g_na * m_inf ^ 3 * s.h * (v - s.e_na)
        i_nap = s.g_nap * s.p * (v - s.e_na)
        i_k = s.g_k * s.n ^ 4 * (v - s.e_k)
        i_ahp = s.g_ahp * ahp_inf * (v - s.e_k)
        i_h = s.g_h * s.r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        dv_val = (-i_na - i_nap - i_k - i_t - i_ahp - i_h - i_l + inp) / s.c_m
        s.v += sub_dt * dv_val
        if s.v >= s.v_threshold
            fired = 1
            s.v = -60.0
            s.ca += 0.2
        end
    end
    s.v = max(-100.0, min(60.0, s.v))
    if ! isfinite(s.v)
        s.v = -60.0
        s.h = 0.6
        s.n = 0.32
    end
    if ! isfinite(s.ca)
        s.ca = 0.05
    end
    s.h = max(0.0, min(1.0, s.h))
    s.n = max(0.0, min(1.0, s.n))
    s.p = max(0.0, min(1.0, s.p))
    s.s = max(0.0, min(1.0, s.s))
    s.r = max(0.0, min(1.0, s.r))
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DCNNeuronState()
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

end # module DcnNeuronAccel
