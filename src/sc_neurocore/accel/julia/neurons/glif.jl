# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for glif

module GlifAccel

export step!, simulate, GLIFNeuronState

mutable struct GLIFNeuronState
    v::Float64
    theta::Float64
    theta_inf::Float64
    i_asc1::Float64
    i_asc2::Float64
    v_rest::Float64
    v_reset::Float64
    tau_m::Float64
    tau_theta::Float64
    tau_asc1::Float64
    tau_asc2::Float64
    a_theta::Float64
    delta_theta::Float64
    r_asc1::Float64
    r_asc2::Float64
    resistance::Float64
    dt::Float64
end

function GLIFNeuronState()
    GLIFNeuronState(-70.0, -50.0, -50.0, 0.0, 0.0, -70.0, -70.0, 10.0, 100.0, 10.0, 200.0, 0.01, 2.0, 1.0, 0.5, 1.0, 1.0)
end

function step!(s::GLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dv = (-(s.v - s.v_rest) + s.resistance * I_ext + s.i_asc1 + s.i_asc2) / s.tau_m * s.dt
        dtheta = (s.theta_inf - s.theta + s.a_theta * (s.v - s.v_rest)) / s.tau_theta * s.dt
        s.i_asc1 *= exp(-s.dt / s.tau_asc1)
        s.i_asc2 *= exp(-s.dt / s.tau_asc2)
        s.v += dv
        s.theta += dtheta
        if s.v >= s.theta
            s.v = s.v_reset
            s.theta += s.delta_theta
            s.i_asc1 += s.r_asc1
            s.i_asc2 += s.r_asc2
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GLIFNeuronState()
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

end # module GlifAccel
