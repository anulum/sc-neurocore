# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for durstewitz_dopamine

module DurstewitzDopamineAccel

export step!, simulate, DurstewitzDopamineNeuronState

mutable struct DurstewitzDopamineNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    g_na::Float64
    g_k::Float64
    g_nmda::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_nmda::Float64
    e_l::Float64
    mg::Float64
    d1_level::Float64
    g_nmda_scale::Float64
    g_k_scale::Float64
    v_shift_na::Float64
    dt::Float64
    v_threshold::Float64
end

function DurstewitzDopamineNeuronState()
    DurstewitzDopamineNeuronState(-65.0, 0.7, 0.2, 45.0, 18.0, 0.5, 0.02, 55.0, -80.0, 0.0, -65.0, 1.0, 0.0, 2.5, 1.5, -5.0, 0.05, -20.0)
end

function step!(s::DurstewitzDopamineNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        d = s.d1_level
        v_sh = d * s.v_shift_na
        m_na_inf = 1.0 / (1.0 + exp(-(s.v + 30.0 + v_sh) / 9.5))
        h_na_inf = 1.0 / (1.0 + exp((s.v + 53.0) / 7.0))
        n_k_inf = 1.0 / (1.0 + exp(-(s.v + 30.0) / 10.0))
        tau_h = 0.5 + 14.0 / (1.0 + exp((s.v + 50.0) / 12.0))
        tau_n = 1.0 + 11.0 / (1.0 + exp((s.v + 40.0) / 10.0))
        s.h_na += (h_na_inf - s.h_na) / tau_h * s.dt
        s.n_k += (n_k_inf - s.n_k) / tau_n * s.dt
        mg_block = 1.0 / (1.0 + s.mg / 3.57 * exp(-0.062 * s.v))
        nmda_g = s.g_nmda * (1.0 + d * (s.g_nmda_scale - 1.0))
        k_g = s.g_k * (1.0 + d * (s.g_k_scale - 1.0))
        i_na = s.g_na * m_na_inf ^ 3 * s.h_na * (s.v - s.e_na)
        i_k = k_g * s.n_k ^ 4 * (s.v - s.e_k)
        i_nmda = nmda_g * mg_block * (s.v - s.e_nmda)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_nmda - i_l + I_ext) * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DurstewitzDopamineNeuronState()
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

end # module DurstewitzDopamineAccel
