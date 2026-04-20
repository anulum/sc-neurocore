# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for traub_miles

module TraubMilesAccel

export step!, simulate, TraubMilesNeuronState

mutable struct TraubMilesNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function TraubMilesNeuronState()
    TraubMilesNeuronState(-67.0, 0.05, 0.6, 0.3, 100.0, 80.0, 0.1, 50.0, -100.0, -67.0, 0.01, -20.0)
end

function step!(s::TraubMilesNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:10
            d = s.v + 54.0
            am = (abs(d) > 1e-06) ? 0.32 * d / (1.0 - exp(-d / 4.0)) : 8.0
            d2 = s.v + 27.0
            bm = (abs(d2) > 1e-06) ? 0.28 * d2 / (exp(d2 / 5.0) - 1.0) : 5.6
            ah = 0.128 * exp(-(s.v + 50.0) / 18.0)
            bh = 4.0 / (1.0 + exp(-(s.v + 27.0) / 5.0))
            d3 = s.v + 52.0
            an = (abs(d3) > 1e-06) ? 0.032 * d3 / (1.0 - exp(-d3 / 5.0)) : 0.32
            bn = 0.5 * exp(-(s.v + 57.0) / 40.0)
            s.m += (am * (1 - s.m) - bm * s.m) * s.dt
            s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
            s.n += (an * (1 - s.n) - bn * s.n) * s.dt
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_l + I_ext) * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = TraubMilesNeuronState()
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

end # module TraubMilesAccel
