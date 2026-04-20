# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for connor_stevens

module ConnorStevensAccel

export step!, simulate, ConnorStevensNeuronState

mutable struct ConnorStevensNeuronState
    v::Float64
    m::Float64
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
    e_a::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function ConnorStevensNeuronState()
    ConnorStevensNeuronState(-68.0, 0.01, 0.99, 0.1, 0.5, 0.1, 120.0, 20.0, 47.7, 0.3, 55.0, -72.0, -75.0, -17.0, 1.0, 0.01, 0.0)
end

function step!(s::ConnorStevensNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in Int(1.0 / max(s.dt:0.001))
            am = (abs(s.v + 29.7) > 1e-06) ? 0.38 * (s.v + 29.7) / (1.0 - exp(-(s.v + 29.7) / 10.0)) : 3.8
            bm = 15.2 * exp(-(s.v + 54.7) / 18.0)
            ah = 0.266 * exp(-(s.v + 48.0) / 20.0)
            bh = 3.8 / (1.0 + exp(-(s.v + 18.0) / 10.0))
            an = (abs(s.v + 45.7) > 1e-06) ? 0.02 * (s.v + 45.7) / (1.0 - exp(-(s.v + 45.7) / 10.0)) : 0.2
            bn = 0.25 * exp(-(s.v + 55.7) / 80.0)
            a_inf = (0.0761 * exp((s.v + 94.22) / 31.84) / (1.0 + exp((s.v + 1.17) / 28.93))) ^ (1.0 / 3.0)
            tau_a = 0.3632 + 1.158 / (1.0 + exp((s.v + 55.96) / 20.12))
            b_inf = 1.0 / (1.0 + exp((s.v + 53.3) / 14.54)) ^ 4
            tau_b = 1.24 + 2.678 / (1.0 + exp((s.v + 50.0) / 16.027))
            s.m += (am * (1 - s.m) - bm * s.m) * s.dt
            s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
            s.n += (an * (1 - s.n) - bn * s.n) * s.dt
            s.a += (a_inf - s.a) / tau_a * s.dt
            s.b += (b_inf - s.b) / tau_b * s.dt
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_a = s.g_a * s.a ^ 3 * s.b * (s.v - s.e_a)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_a - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ConnorStevensNeuronState()
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

end # module ConnorStevensAccel
