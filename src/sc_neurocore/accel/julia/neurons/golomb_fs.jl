# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 kernel for the Golomb et al. 2007 fast-spiking neuron

module GolombFsAccel

export step!, simulate, GolombFSNeuronState

mutable struct GolombFSNeuronState
    v::Float64
    h::Float64
    n::Float64
    p::Float64
    g_na::Float64
    g_kd::Float64
    g_kv3::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function GolombFSNeuronState()
    GolombFSNeuronState(-65.0, 0.9, 0.1, 0.0, 112.5, 225.0, 150.0, 0.25, 50.0, -90.0, -70.0, 1.0, 0.01, -20.0)
end

# Return (dV, dh, dn, dp) of the four-state system at one consistent state.
@inline function derivatives(s::GolombFSNeuronState, v, h, n, p, current)
    m_inf = 1.0 / (1.0 + exp(-(v + 24.0) / 11.5))
    h_inf = 1.0 / (1.0 + exp((v + 58.3) / 6.7))
    tau_h = 0.5 + 14.0 / (1.0 + exp((v + 60.0) / 12.0))
    n_inf = 1.0 / (1.0 + exp(-(v + 12.4) / 6.8))
    tau_n = 0.087 + 11.4 / (1.0 + exp((v + 14.6) / 8.6))
    p_inf = 1.0 / (1.0 + exp(-(v + 3.0) / 8.0))
    tau_p = 0.1 + 4.0 / (1.0 + exp((v + 25.0) / 10.0))
    dh = (h_inf - h) / tau_h
    dn = (n_inf - n) / tau_n
    dp = (p_inf - p) / tau_p
    i_na = s.g_na * m_inf * m_inf * m_inf * h * (v - s.e_na)
    i_kd = s.g_kd * n * n * n * n * (v - s.e_k)
    i_kv3 = s.g_kv3 * p * p * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = (-i_na - i_kd - i_kv3 - i_l + current) / s.c_m
    return (dv, dh, dn, dp)
end

# One classical RK4 increment of (V, h, n, p), holding `current` constant.
@inline function rk4_substep(s::GolombFSNeuronState, st::NTuple{4,Float64}, current::Float64)
    dt = s.dt
    k1 = derivatives(s, st[1], st[2], st[3], st[4], current)
    k2 = derivatives(s, st[1] + 0.5 * dt * k1[1], st[2] + 0.5 * dt * k1[2], st[3] + 0.5 * dt * k1[3], st[4] + 0.5 * dt * k1[4], current)
    k3 = derivatives(s, st[1] + 0.5 * dt * k2[1], st[2] + 0.5 * dt * k2[2], st[3] + 0.5 * dt * k2[3], st[4] + 0.5 * dt * k2[4], current)
    k4 = derivatives(s, st[1] + dt * k3[1], st[2] + dt * k3[2], st[3] + dt * k3[3], st[4] + dt * k3[4], current)
    return ntuple(i -> st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 4)
end

function step!(s::GolombFSNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    st = (s.v, s.h, s.n, s.p)
    for _ in 1:10
        st = rk4_substep(s, st, I_ext)
    end
    s.v, s.h, s.n, s.p = st
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GolombFSNeuronState()
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

end # module GolombFsAccel
