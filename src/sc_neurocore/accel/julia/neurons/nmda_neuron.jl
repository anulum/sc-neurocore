# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for nmda_neuron

module NmdaNeuronAccel

export step!, simulate, NMDANeuronState

mutable struct NMDANeuronState
    v::Float64
    h::Float64
    n::Float64
    s_nmda::Float64
    g_na::Float64
    g_k::Float64
    g_nmda::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_nmda::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    mg_conc::Float64
    tau_rise::Float64
    tau_decay::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    _sub_steps::Float64
end

function NMDANeuronState()
    NMDANeuronState(-65.0, 0.6, 0.32, 0.0, 35.0, 9.0, 0.5, 0.1, 55.0, -90.0, 0.0, -65.0, 1.0, 5.0, 1.0, 10.0, 100.0, 0.5, -20.0, 1.0, 0.0)
end

function step!(s::NMDANeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        inp = s.gain * I_ext
        sub_dt = s.dt / s._sub_steps
        fired = 0
        drive = (inp > 0.0) ? inp / (inp + 5.0) : 0.0
        tau = (drive > s.s_nmda) ? s.tau_rise : s.tau_decay
        ds = (drive - s.s_nmda) / tau
        s.s_nmda += s.dt * ds
        s.s_nmda = max(0.0, min(1.0, s.s_nmda))
        for _ in 1:s._sub_steps
            v = s.v
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)
            alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
            mg_block = 1.0 / (1.0 + s.mg_conc / 3.57 * exp(-0.062 * v))
            s.h += sub_dt * s.phi * (alpha_h * (1.0 - s.h) - beta_h * s.h)
            s.n += sub_dt * s.phi * (alpha_n * (1.0 - s.n) - beta_n * s.n)
            i_na = s.g_na * m_inf ^ 3 * s.h * (v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (v - s.e_k)
            i_nmda = s.g_nmda * s.s_nmda * mg_block * (v - s.e_nmda)
            i_l = s.g_l * (v - s.e_l)
            dv = (-i_na - i_k - i_nmda - i_l + inp) / s.c_m
            s.v += sub_dt * dv
            if s.v >= s.v_threshold
                fired = 1
                s.v = -65.0
            end
        end
        s.v = max(-100.0, min(60.0, s.v))
        if ! isfinite(s.v)
            s.v = -65.0
            s.h = 0.6
            s.n = 0.32
        end
        if ! isfinite(s.s_nmda)
            s.s_nmda = 0.0
        end
        s.h = max(0.0, min(1.0, s.h))
        s.n = max(0.0, min(1.0, s.n))
        return fired
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = NMDANeuronState()
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

end # module NmdaNeuronAccel
