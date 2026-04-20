# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for cerebellar_basket_neuron

module CerebellarBasketNeuronAccel

export step!, simulate, CerebellarBasketNeuronState

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

function step!(s::CerebellarBasketNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    for _ in 1:n_sub
        am = _safe_rate(0.1, 35.0, s.v, 10.0, 1.0)
        bm = 4.0 * exp(-(s.v + 60.0) / 18.0)
        m_inf = am / (am + bm)
        ah = 0.07 * exp(-(s.v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + exp(-(s.v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, s.v, 10.0, 0.1)
        bn = 0.125 * exp(-(s.v + 44.0) / 80.0)
        s.h += s.phi * (ah * (1.0 - s.h) - bh * s.h) * s.dt
        s.n += s.phi * (an * (1.0 - s.n) - bn * s.n) * s.dt
        a_inf = 1.0 / (1.0 + exp(-(s.v + 45.0) / 15.0))
        b_inf = 1.0 / (1.0 + exp((s.v + 75.0) / 8.0))
        s.a += s.phi * (a_inf - s.a) / 5.0 * s.dt
        s.b += (b_inf - s.b) / 50.0 * s.dt
        q_inf = s.ca / (s.ca + 0.2)
        i_ca_entry = (s.v > -20.0) ? 0.01 * (s.v + 20.0) : 0.0
        s.ca += (-s.ca / 80.0 + i_ca_entry) * s.dt
        s.ca = max(0.0, s.ca)
        i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
        i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
        i_a = s.g_a * s.a ^ 3 * s.b * (s.v - s.e_k)
        i_kca = s.g_kca * q_inf * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_a - i_kca - i_l + I_ext) / s.c_m * s.dt
    end
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
