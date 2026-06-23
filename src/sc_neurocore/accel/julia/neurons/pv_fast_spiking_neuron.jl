# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 kernel for the PV+ fast-spiking neuron

module PvFastSpikingNeuronAccel

export step!, simulate, PVFastSpikingNeuronState

mutable struct PVFastSpikingNeuronState
    v::Float64
    h::Float64
    n::Float64
    p::Float64
    g_na::Float64
    g_k::Float64
    g_kv3::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function PVFastSpikingNeuronState()
    PVFastSpikingNeuronState(-65.0, 0.8, 0.1, 0.0, 35.0, 9.0, 5.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.01, -20.0)
end

# Wang-Buzsáki activation rate with the closed-form L'Hôpital limit (a*k) within
# 1e-7 of its x/(1-exp(-x/k)) removable singularity; matches the other kernels.
@inline function safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    abs(d) < 1e-7 ? fallback : a * d / (1.0 - exp(-d / k))
end

# Return (dV, dh, dn, dp) of the four-state system at one consistent state.
@inline function derivatives(s::PVFastSpikingNeuronState, v, h, n, p, current)
    am = safe_rate(0.1, 35.0, v, 10.0, 1.0)
    bm = 4.0 * exp(-(v + 60.0) / 18.0)
    m_inf = am / (am + bm)
    ah = 0.07 * exp(-(v + 58.0) / 20.0)
    bh = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
    an = safe_rate(0.01, 34.0, v, 10.0, 0.1)
    bn = 0.125 * exp(-(v + 44.0) / 80.0)
    p_inf = 1.0 / (1.0 + exp(-(v + 10.0) / 10.0))
    dh = s.phi * (ah * (1.0 - h) - bh * h)
    dn = s.phi * (an * (1.0 - n) - bn * n)
    dp = s.phi * (p_inf - p)
    i_na = s.g_na * m_inf * m_inf * m_inf * h * (v - s.e_na)
    i_k = s.g_k * n * n * n * n * (v - s.e_k)
    i_kv3 = s.g_kv3 * p * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = (-i_na - i_k - i_kv3 - i_l + current) / s.c_m
    return (dv, dh, dn, dp)
end

# One classical RK4 increment of (V, h, n, p), holding `current` constant.
@inline function rk4_substep(s::PVFastSpikingNeuronState, st::NTuple{4,Float64}, current::Float64)
    dt = s.dt
    k1 = derivatives(s, st[1], st[2], st[3], st[4], current)
    k2 = derivatives(s, st[1] + 0.5 * dt * k1[1], st[2] + 0.5 * dt * k1[2], st[3] + 0.5 * dt * k1[3], st[4] + 0.5 * dt * k1[4], current)
    k3 = derivatives(s, st[1] + 0.5 * dt * k2[1], st[2] + 0.5 * dt * k2[2], st[3] + 0.5 * dt * k2[3], st[4] + 0.5 * dt * k2[4], current)
    k4 = derivatives(s, st[1] + dt * k3[1], st[2] + dt * k3[2], st[3] + dt * k3[3], st[4] + dt * k3[4], current)
    return ntuple(i -> st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 4)
end

function step!(s::PVFastSpikingNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    n_sub = max(1, Int(round(0.5 / max(s.dt, 0.001))))
    st = (s.v, s.h, s.n, s.p)
    for _ in 1:n_sub
        st = rk4_substep(s, st, I_ext)
    end
    s.v, s.h, s.n, s.p = st
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PVFastSpikingNeuronState()
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

end # module PvFastSpikingNeuronAccel
