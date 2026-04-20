# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for atype_k_neuron

module AtypeKNeuronAccel

export step!, simulate, ATypeKNeuronState

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
    _sub_steps::Float64
end

function ATypeKNeuronState()
    ATypeKNeuronState(-65.0, 0.6, 0.32, 0.1, 0.8, 35.0, 9.0, 8.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.5, -20.0, 1.0, 0.0)
end

function step!(s::ATypeKNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
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
            a_inf = 1.0 / (1.0 + exp(-(v + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + exp((v + 70.0) / 6.0))
            s.h += sub_dt * s.phi * (alpha_h * (1.0 - s.h) - beta_h * s.h)
            s.n += sub_dt * s.phi * (alpha_n * (1.0 - s.n) - beta_n * s.n)
            s.a += sub_dt * (a_inf - s.a) / 2.0
            s.b += sub_dt * (b_inf - s.b) / 50.0
            i_na = s.g_na * m_inf ^ 3 * s.h * (v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (v - s.e_k)
            i_a = s.g_a * s.a ^ 3 * s.b * (v - s.e_k)
            i_l = s.g_l * (v - s.e_l)
            dv = (-i_na - i_k - i_a - i_l + inp) / s.c_m
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
        s.h = max(0.0, min(1.0, s.h))
        s.n = max(0.0, min(1.0, s.n))
        s.a = max(0.0, min(1.0, s.a))
        s.b = max(0.0, min(1.0, s.b))
        return fired
    catch _e
        return 0
    end
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
