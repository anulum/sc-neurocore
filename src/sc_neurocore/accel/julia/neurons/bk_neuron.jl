# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for bk_neuron

module BkNeuronAccel

export step!, simulate, BKNeuronState

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
    _sub_steps::Float64
end

function BKNeuronState()
    BKNeuronState(-65.0, 0.6, 0.32, 0.0, 35.0, 9.0, 3.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 50.0, 0.5, -20.0, 1.0, 0.0)
end

function step!(s::BKNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
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
            v_half_bk = 10.0 - 30.0 * (s.ca / (s.ca + 0.5))
            bk_inf = 1.0 / (1.0 + exp(-(v - v_half_bk) / 15.0))
            s.ca += sub_dt * (-s.ca / s.tau_ca)
            s.h += sub_dt * s.phi * (alpha_h * (1.0 - s.h) - beta_h * s.h)
            s.n += sub_dt * s.phi * (alpha_n * (1.0 - s.n) - beta_n * s.n)
            i_na = s.g_na * m_inf ^ 3 * s.h * (v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (v - s.e_k)
            i_bk = s.g_bk * bk_inf * (v - s.e_k)
            i_l = s.g_l * (v - s.e_l)
            dv = (-i_na - i_k - i_bk - i_l + inp) / s.c_m
            s.v += sub_dt * dv
            if s.v >= s.v_threshold
                fired = 1
                s.v = -65.0
                s.ca += 0.3
            end
        end
        s.v = max(-100.0, min(60.0, s.v))
        if ! isfinite(s.v)
            s.v = -65.0
            s.h = 0.6
            s.n = 0.32
        end
        if ! isfinite(s.ca)
            s.ca = 0.0
        end
        s.h = max(0.0, min(1.0, s.h))
        s.n = max(0.0, min(1.0, s.n))
        s.ca = max(0.0, s.ca)
        return fired
    catch _e
        return 0
    end
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
