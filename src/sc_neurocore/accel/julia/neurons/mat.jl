# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mat

module MatAccel

export step!, simulate, MATNeuronState

mutable struct MATNeuronState
    v::Float64
    theta1::Float64
    theta2::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold_base::Float64
    tau_m::Float64
    tau_1::Float64
    tau_2::Float64
    h1::Float64
    h2::Float64
    resistance::Float64
    dt::Float64
end

function MATNeuronState()
    MATNeuronState(-70.0, 0.0, 0.0, -70.0, -70.0, -50.0, 10.0, 10.0, 200.0, 5.0, 3.0, 1.0, 1.0)
end

function step!(s::MATNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-(s.v - s.v_rest) + s.resistance * I_ext) / s.tau_m * s.dt
        s.theta1 *= exp(-s.dt / s.tau_1)
        s.theta2 *= exp(-s.dt / s.tau_2)
        threshold = s.v_threshold_base + s.theta1 + s.theta2
        if s.v >= threshold
            s.v = s.v_reset
            s.theta1 += s.h1
            s.theta2 += s.h2
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MATNeuronState()
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

end # module MatAccel
