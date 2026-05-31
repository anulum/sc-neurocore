# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ermentrout_kopell_map_neuron

module ErmentroutKopellMapNeuronAccel

export step!, simulate, validate, ErmentroutKopellMapNeuronState

mutable struct ErmentroutKopellMapNeuronState
    theta::Float64
    dt::Float64
    gain::Float64
    theta_threshold::Float64
end

function ErmentroutKopellMapNeuronState()
    ErmentroutKopellMapNeuronState(0.0, 0.1, 1.0, pi)
end

function validate(s::ErmentroutKopellMapNeuronState)::Bool
    return isfinite(s.theta) &&
        isfinite(s.dt) &&
        s.dt > 0.0 &&
        isfinite(s.gain) &&
        isfinite(s.theta_threshold)
end

function step!(s::ErmentroutKopellMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end

    inp = s.gain * I_ext
    if !isfinite(inp)
        return -1
    end
    theta_prev = s.theta
    cos_theta = cos(s.theta)
    d_theta = 1.0 - cos_theta + (1.0 + cos_theta) * inp
    theta_next = s.theta + s.dt * d_theta
    if !isfinite(d_theta) || !isfinite(theta_next)
        return -1
    end
    fired = (theta_next >= s.theta_threshold && theta_prev < s.theta_threshold) ? 1 : 0
    s.theta = mod(theta_next, 2.0 * pi)
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ErmentroutKopellMapNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.theta
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ErmentroutKopellMapNeuronAccel
