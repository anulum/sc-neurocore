# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 kernel for the Pospischil et al. 2008 neuron

module PospischilAccel

export step!, simulate, PospischilNeuronState

mutable struct PospischilNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    p::Float64
    g_na::Float64
    g_kd::Float64
    g_m::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    vt::Float64
    dt::Float64
    v_threshold::Float64
end

function PospischilNeuronState()
    PospischilNeuronState(-70.0, 0.05, 0.6, 0.3, 0.0, 50.0, 5.0, 0.07, 0.1, 50.0, -90.0, -70.0, 1.0, -56.2, 0.025, -20.0)
end

# Traub-Miles activation rate with the closed-form L'Hôpital limit within 1e-6 of
# its x/(exp(±x/k)-1) removable singularity; matches the Python/Rust/Go/Mojo kernels.
@inline function alpha_singular(num::Float64, slope::Float64, limit::Float64)
    abs(num) < 1e-6 ? limit : num / (exp(num / slope) - 1.0)
end

# Return (dV, dm, dh, dn, dp) of the five-state system at one consistent state.
@inline function derivatives(s::PospischilNeuronState, v, m, h, n, p, current)
    dv_vt = v - s.vt
    am = -0.32 * alpha_singular(dv_vt - 13.0, -4.0, -4.0)
    bm = 0.28 * alpha_singular(dv_vt - 40.0, 5.0, 5.0)
    ah = 0.128 * exp(-(dv_vt - 17.0) / 18.0)
    bh = 4.0 / (1.0 + exp(-(dv_vt - 40.0) / 5.0))
    an = -0.032 * alpha_singular(dv_vt - 15.0, -5.0, -5.0)
    bn = 0.5 * exp(-(dv_vt - 10.0) / 40.0)
    p_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
    tau_p = 608.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    dp = (p_inf - p) / tau_p
    i_na = s.g_na * m * m * m * h * (v - s.e_na)
    i_kd = s.g_kd * n * n * n * n * (v - s.e_k)
    i_m = s.g_m * p * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = (-i_na - i_kd - i_m - i_l + current) / s.c_m
    return (dv, dm, dh, dn, dp)
end

# One classical RK4 increment of (V, m, h, n, p), holding `current` constant.
@inline function rk4_substep(s::PospischilNeuronState, st::NTuple{5,Float64}, current::Float64)
    dt = s.dt
    k1 = derivatives(s, st[1], st[2], st[3], st[4], st[5], current)
    k2 = derivatives(s, st[1] + 0.5 * dt * k1[1], st[2] + 0.5 * dt * k1[2], st[3] + 0.5 * dt * k1[3], st[4] + 0.5 * dt * k1[4], st[5] + 0.5 * dt * k1[5], current)
    k3 = derivatives(s, st[1] + 0.5 * dt * k2[1], st[2] + 0.5 * dt * k2[2], st[3] + 0.5 * dt * k2[3], st[4] + 0.5 * dt * k2[4], st[5] + 0.5 * dt * k2[5], current)
    k4 = derivatives(s, st[1] + dt * k3[1], st[2] + dt * k3[2], st[3] + dt * k3[3], st[4] + dt * k3[4], st[5] + dt * k3[5], current)
    return ntuple(i -> st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 5)
end

function step!(s::PospischilNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    st = (s.v, s.m, s.h, s.n, s.p)
    for _ in 1:4
        st = rk4_substep(s, st, I_ext)
    end
    s.v, s.m, s.h, s.n, s.p = st
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PospischilNeuronState()
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

end # module PospischilAccel
