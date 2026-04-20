# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pv_fast_spiking_neuron

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

function step!(s::PVFastSpikingNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
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
            p_inf = 1.0 / (1.0 + exp(-(s.v + 10.0) / 10.0))
            s.p += s.phi * (p_inf - s.p) / 1.0 * s.dt
            i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_kv3 = s.g_kv3 * s.p * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_kv3 - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PVFastSpikingNeuronState()
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

end # module PvFastSpikingNeuronAccel
