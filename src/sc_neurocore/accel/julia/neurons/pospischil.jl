# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pospischil

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

function step!(s::PospischilNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:4
            dv = s.v - s.vt
            am = -0.32 * (dv - 13.0) / (exp(-(dv - 13.0) / 4.0) - 1.0 + 1e-12)
            bm = 0.28 * (dv - 40.0) / (exp((dv - 40.0) / 5.0) - 1.0 + 1e-12)
            ah = 0.128 * exp(-(dv - 17.0) / 18.0)
            bh = 4.0 / (1.0 + exp(-(dv - 40.0) / 5.0))
            an = -0.032 * (dv - 15.0) / (exp(-(dv - 15.0) / 5.0) - 1.0 + 1e-12)
            bn = 0.5 * exp(-(dv - 10.0) / 40.0)
            p_inf = 1.0 / (1.0 + exp(-(s.v + 35.0) / 10.0))
            tau_p = 608.0 / (3.3 * exp((s.v + 35.0) / 20.0) + exp(-(s.v + 35.0) / 20.0))
            s.m += (am * (1 - s.m) - bm * s.m) * s.dt
            s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
            s.n += (an * (1 - s.n) - bn * s.n) * s.dt
            s.p += (p_inf - s.p) / tau_p * s.dt
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
            i_kd = s.g_kd * s.n ^ 4 * (s.v - s.e_k)
            i_m = s.g_m * s.p * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_kd - i_m - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PospischilNeuronState()
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

end # module PospischilAccel
