# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for loihi2

module Loihi2Accel

export step!, simulate, Loihi2NeuronState

mutable struct Loihi2NeuronState
    s1::Float64
    s2::Float64
    s3::Float64
    tau1::Float64
    tau2::Float64
    tau3::Float64
    w12::Float64
    w13::Float64
    w23::Float64
    s1_threshold::Float64
    s1_reset::Float64
    s3_incr::Float64
end

function Loihi2NeuronState()
    Loihi2NeuronState(0.0, 0.0, 0.0, 10.0, 5.0, 50.0, 1.0, 0.0, 0.0, 1000.0, 0.0, 10.0)
end

function step!(s::Loihi2NeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.s3 -= s.s3 // s.tau3
        s.s2 = s.s2 - s.s2 // s.tau2 + weighted_input + s.w23 * s.s3
        s.s1 = s.s1 - s.s1 // s.tau1 + s.w12 * s.s2 + s.w13 * s.s3
        if s.s1 >= s.s1_threshold
            s.s1 = s.s1_reset
            s.s3 += s.s3_incr
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = Loihi2NeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.s1
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module Loihi2Accel
