# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for wang_buzsaki

module WangBuzsakiAccel

export step!, simulate, WangBuzsakiNeuronState

mutable struct WangBuzsakiNeuronState
    v::Float64
    h::Float64
    n::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function WangBuzsakiNeuronState()
    WangBuzsakiNeuronState(-65.0, 0.8, 0.1, 35.0, 9.0, 0.1, 55.0, -90.0, -65.0, 1.0, 5.0, 0.01, -20.0)
end

function step!(s::WangBuzsakiNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in Int(0.5 / max(s.dt:0.001))
            alpha_m = (abs(s.v + 35.0) > 1e-06) ? 0.1 * (s.v + 35.0) / (1.0 - exp(-(s.v + 35.0) / 10.0)) : 1.0
            beta_m = 4.0 * exp(-(s.v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)
            alpha_h = 0.07 * exp(-(s.v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + exp(-(s.v + 28.0) / 10.0))
            alpha_n = (abs(s.v + 34.0) > 1e-06) ? 0.01 * (s.v + 34.0) / (1.0 - exp(-(s.v + 34.0) / 10.0)) : 0.1
            beta_n = 0.125 * exp(-(s.v + 44.0) / 80.0)
            s.h += s.phi * (alpha_h * (1 - s.h) - beta_h * s.h) * s.dt
            s.n += s.phi * (alpha_n * (1 - s.n) - beta_n * s.n) * s.dt
            i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = WangBuzsakiNeuronState()
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

end # module WangBuzsakiAccel
