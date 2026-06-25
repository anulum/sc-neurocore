# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 kernel for the VIP irregular-spiking neuron

module VipNeuronAccel

export step!, simulate, VIPNeuronState

mutable struct VIPNeuronState
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
    dt::Float64
    v_threshold::Float64
end

function VIPNeuronState()
    VIPNeuronState(-65.0, 0.8, 0.1, 0.0, 0.9, 35.0, 6.0, 8.0, 0.01, 55.0, -90.0, -65.0, 0.5, 0.025, -20.0)
end

# Return (dV, dh, dn, da, db) of the five-state system at one consistent state.
@inline function derivatives(s::VIPNeuronState, v, h, n, a, b, current)
    m_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 9.5))
    h_inf = 1.0 / (1.0 + exp((v + 53.0) / 7.0))
    tau_h = 0.37 + 2.78 / (1.0 + exp((v + 40.5) / 6.0))
    n_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 10.0))
    tau_n = 0.37 + 1.85 / (1.0 + exp((v + 27.0) / 15.0))
    a_inf = 1.0 / (1.0 + exp(-(v + 50.0) / 20.0))
    b_inf = 1.0 / (1.0 + exp((v + 78.0) / 6.0))
    dh = (h_inf - h) / tau_h
    dn = (n_inf - n) / tau_n
    da = (a_inf - a) / 5.0
    db = (b_inf - b) / 50.0
    i_na = s.g_na * m_inf * m_inf * m_inf * h * (v - s.e_na)
    i_k = s.g_k * n * n * n * n * (v - s.e_k)
    i_a = s.g_a * a * a * a * b * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = (-i_na - i_k - i_a - i_l + current) / s.c_m
    return (dv, dh, dn, da, db)
end

# One classical RK4 increment of (V, h, n, a, b), holding `current` constant.
@inline function rk4_substep(s::VIPNeuronState, st::NTuple{5,Float64}, current::Float64)
    dt = s.dt
    k1 = derivatives(s, st[1], st[2], st[3], st[4], st[5], current)
    k2 = derivatives(s, st[1] + 0.5 * dt * k1[1], st[2] + 0.5 * dt * k1[2], st[3] + 0.5 * dt * k1[3], st[4] + 0.5 * dt * k1[4], st[5] + 0.5 * dt * k1[5], current)
    k3 = derivatives(s, st[1] + 0.5 * dt * k2[1], st[2] + 0.5 * dt * k2[2], st[3] + 0.5 * dt * k2[3], st[4] + 0.5 * dt * k2[4], st[5] + 0.5 * dt * k2[5], current)
    k4 = derivatives(s, st[1] + dt * k3[1], st[2] + dt * k3[2], st[3] + dt * k3[3], st[4] + dt * k3[4], st[5] + dt * k3[5], current)
    return ntuple(i -> st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 5)
end

function step!(s::VIPNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    st = (s.v, s.h, s.n, s.a, s.b)
    for _ in 1:4
        st = rk4_substep(s, st, I_ext)
    end
    s.v, s.h, s.n, s.a, s.b = st
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = VIPNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module VipNeuronAccel
