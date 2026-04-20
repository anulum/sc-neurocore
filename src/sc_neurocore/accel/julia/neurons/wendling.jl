# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for wendling

module WendlingAccel

export step!, simulate, WendlingNeuronState

mutable struct WendlingNeuronState
    y0::Float64
    y5::Float64
    y1::Float64
    y6::Float64
    y2::Float64
    y7::Float64
    y3::Float64
    y8::Float64
    y4::Float64
    y9::Float64
    a_exc::Float64
    b_fast::Float64
    g_slow::Float64
    a_rate::Float64
    b_rate::Float64
    g_rate::Float64
    c::Float64
    e0::Float64
    v0::Float64
    r::Float64
    dt::Float64
end

function WendlingNeuronState()
    WendlingNeuronState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.25, 22.0, 10.0, 100.0, 500.0, 20.0, 135.0, 2.5, 6.0, 0.56, 0.001)
end

function _sigmoid(s::WendlingNeuronState, x)
    return 2.0 * s.e0 / (1.0 + exp(s.r * (s.v0 - x)))
end

function step!(s::WendlingNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        sig_1_2_3_4 = s._sigmoid(s.y1 - s.y2 - s.y3)
        sig_0 = s._sigmoid(s.c * 0.8 * s.y0)
        sig_fast = s._sigmoid(s.c * 0.25 * s.y0)
        sig_slow = s._sigmoid(s.c * 0.1 * s.y0)
        dy0 = s.y5
        dy5 = s.a_exc * s.a_rate * sig_1_2_3_4 - 2 * s.a_rate * s.y5 - s.a_rate ^ 2 * s.y0
        dy1 = s.y6
        dy6 = s.a_exc * s.a_rate * (p_ext + s.c * 0.8 * sig_0) - 2 * s.a_rate * s.y6 - s.a_rate ^ 2 * s.y1
        dy2 = s.y7
        dy7 = s.b_fast * s.b_rate * s.c * 0.25 * sig_fast - 2 * s.b_rate * s.y7 - s.b_rate ^ 2 * s.y2
        dy3 = s.y8
        dy8 = s.g_slow * s.g_rate * s.c * 0.1 * sig_slow - 2 * s.g_rate * s.y8 - s.g_rate ^ 2 * s.y3
        s.y0 += dy0 * s.dt
        s.y5 += dy5 * s.dt
        s.y1 += dy1 * s.dt
        s.y6 += dy6 * s.dt
        s.y2 += dy2 * s.dt
        s.y7 += dy7 * s.dt
        s.y3 += dy3 * s.dt
        s.y8 += dy8 * s.dt
        return s.y1 - s.y2 - s.y3
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = WendlingNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.y0
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module WendlingAccel
