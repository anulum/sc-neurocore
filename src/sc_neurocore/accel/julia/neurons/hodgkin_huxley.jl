# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hodgkin_huxley

module HodgkinHuxleyAccel

export step!, simulate, HodgkinHuxleyNeuronState

mutable struct HodgkinHuxleyNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    c_m::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function HodgkinHuxleyNeuronState()
    HodgkinHuxleyNeuronState(-65.0, 0.05, 0.6, 0.32, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 0.01, 0.0)
end

function _alpha_m(s::HodgkinHuxleyNeuronState, v)
    d = v + 40.0
    if abs(d) < 1e-07
        return 1.0
    end
    return 0.1 * d / (1.0 - exp(-d / 10.0))
end

function _beta_m(s::HodgkinHuxleyNeuronState, v)
    return Float64(4.0 * exp(-(v + 65.0) / 18.0))
end

function _alpha_h(s::HodgkinHuxleyNeuronState, v)
    return Float64(0.07 * exp(-(v + 65.0) / 20.0))
end

function _beta_h(s::HodgkinHuxleyNeuronState, v)
    return Float64(1.0 / (1.0 + exp(-(v + 35.0) / 10.0)))
end

function _alpha_n(s::HodgkinHuxleyNeuronState, v)
    d = v + 55.0
    if abs(d) < 1e-07
        return 0.1
    end
    return 0.01 * d / (1.0 - exp(-d / 10.0))
end

function _beta_n(s::HodgkinHuxleyNeuronState, v)
    return Float64(0.125 * exp(-(v + 65.0) / 80.0))
end

function step!(s::HodgkinHuxleyNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:round(1.0 / s.dt)
            (am, bm) = (s._alpha_m(s.v), s._beta_m(s.v))
            (ah, bh) = (s._alpha_h(s.v), s._beta_h(s.v))
            (an, bn) = (s._alpha_n(s.v), s._beta_n(s.v))
            s.m += (am * (1 - s.m) - bm * s.m) * s.dt
            s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
            s.n += (an * (1 - s.n) - bn * s.n) * s.dt
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
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
    s = HodgkinHuxleyNeuronState()
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

end # module HodgkinHuxleyAccel
