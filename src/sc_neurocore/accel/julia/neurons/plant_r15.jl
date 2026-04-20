# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for plant_r15

module PlantR15Accel

export step!, simulate, PlantR15NeuronState

mutable struct PlantR15NeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_ca::Float64
    g_l::Float64
    g_kca::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    k_ca::Float64
    tau_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function PlantR15NeuronState()
    PlantR15NeuronState(-50.0, 0.05, 0.6, 0.3, 0.1, 4.0, 0.3, 0.004, 0.003, 0.03, 30.0, -75.0, 140.0, -40.0, 1.0, 0.0085, 500.0, 0.05, -10.0)
end

function step!(s::PlantR15NeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:5
            am = 0.1 * (50.0 + s.v) / (1.0 - exp(-(50.0 + s.v) / 10.0) + 1e-12)
            bm = 4.0 * exp(-(75.0 + s.v) / 18.0)
            ah = 0.07 * exp(-(s.v + 50.0) / 20.0)
            bh = 1.0 / (1.0 + exp(-(20.0 + s.v) / 10.0))
            an = 0.01 * (55.0 + s.v) / (1.0 - exp(-(55.0 + s.v) / 10.0) + 1e-12)
            bn = 0.125 * exp(-(65.0 + s.v) / 80.0)
            s.m += (am * (1 - s.m) - bm * s.m) * s.dt
            s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
            s.n += (an * (1 - s.n) - bn * s.n) * s.dt
            m_ca_inf = 1.0 / (1.0 + exp(-(s.v + 25.0) / 5.0))
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_ca = s.g_ca * m_ca_inf ^ 2 * (s.v - s.e_ca)
            i_kca = s.g_kca * s.ca / (0.5 + s.ca) * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_ca - i_kca - i_l + I_ext) / s.c_m * s.dt
            s.ca += (-s.k_ca * i_ca - s.ca / s.tau_ca) * s.dt
            s.ca = max(s.ca, 0.0)
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PlantR15NeuronState()
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

end # module PlantR15Accel
