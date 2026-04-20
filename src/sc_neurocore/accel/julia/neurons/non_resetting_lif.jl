# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for non_resetting_lif

module NonResettingLifAccel

export step!, simulate, NonResettingLIFNeuronState

mutable struct NonResettingLIFNeuronState
    v::Float64
    theta::Float64
    v_rest::Float64
    theta_rest::Float64
    delta_theta::Float64
    tau_m::Float64
    tau_theta::Float64
    r_m::Float64
    dt::Float64
end

function NonResettingLIFNeuronState()
    NonResettingLIFNeuronState(-65.0, -50.0, -65.0, -50.0, 5.0, 10.0, 50.0, 1.0, 0.1)
end

function step!(s::NonResettingLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-(s.v - s.v_rest) + s.r_m * I_ext) / s.tau_m * s.dt
        s.theta += -(s.theta - s.theta_rest) / s.tau_theta * s.dt
        if s.v >= s.theta
            s.theta += s.delta_theta
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = NonResettingLIFNeuronState()
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

end # module NonResettingLifAccel
