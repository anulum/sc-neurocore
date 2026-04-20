# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mihalas_niebur

module MihalasNieburAccel

export step!, simulate, MihalasNieburNeuronState

mutable struct MihalasNieburNeuronState
    v::Float64
    theta::Float64
    i1::Float64
    i2::Float64
    v_rest::Float64
    v_reset::Float64
    theta_reset::Float64
    theta_inf::Float64
    tau_v::Float64
    tau_theta::Float64
    tau_1::Float64
    tau_2::Float64
    a::Float64
    b::Float64
    r1::Float64
    r2::Float64
    dt::Float64
end

function MihalasNieburNeuronState()
    MihalasNieburNeuronState(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 100.0, 10.0, 200.0, 0.0, 0.0, 0.0, 0.0, 1.0)
end

function step!(s::MihalasNieburNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dv = (-(s.v - s.v_rest) + s.i1 + s.i2 + I_ext) / s.tau_v * s.dt
        dtheta = (s.theta_inf - s.theta + s.a * (s.v - s.v_rest)) / s.tau_theta * s.dt
        di1 = -s.i1 / s.tau_1 * s.dt
        di2 = -s.i2 / s.tau_2 * s.dt
        s.v += dv
        s.theta += dtheta
        s.i1 += di1
        s.i2 += di2
        if s.v >= s.theta
            s.v = s.v_reset
            s.theta = max(s.theta, s.theta_reset)
            s.i1 += s.r1
            s.i2 += s.r2
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MihalasNieburNeuronState()
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

end # module MihalasNieburAccel
