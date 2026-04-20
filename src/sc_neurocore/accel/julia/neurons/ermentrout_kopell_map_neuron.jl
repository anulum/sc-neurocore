# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ermentrout_kopell_map_neuron

module ErmentroutKopellMapNeuronAccel

export step!, simulate, ErmentroutKopellMapNeuronState

mutable struct ErmentroutKopellMapNeuronState
    theta::Float64
    dt::Float64
    gain::Float64
    theta_threshold::Float64
end

function ErmentroutKopellMapNeuronState()
    ErmentroutKopellMapNeuronState(0.0, 0.1, 1.0, 0.0)
end

function step!(s::ErmentroutKopellMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        inp = s.gain * I_ext
        theta_prev = s.theta
        d_theta = 1.0 - cos(s.theta) + (1.0 + cos(s.theta)) * inp
        s.theta += s.dt * d_theta
        fired = (s.theta >= s.theta_threshold && theta_prev < s.theta_threshold) ? 1 : 0
        two_pi = 2.0 * pi
        if s.theta >= two_pi
            s.theta -= two_pi
        end
        if s.theta < 0.0
            s.theta += two_pi
        end
        if ! isfinite(s.theta)
            s.theta = 0.0
        end
        return fired
    catch _e
        return 0
    end
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
