# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for theta

module ThetaAccel

export step!, simulate, ThetaNeuronState

mutable struct ThetaNeuronState
    theta::Float64
    dt::Float64
end

function ThetaNeuronState()
    ThetaNeuronState(0.0, 0.01)
end

function step!(s::ThetaNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        theta_prev = s.theta
        dtheta = (1.0 - cos(s.theta) + (1.0 + cos(s.theta)) * I_ext) * s.dt
        s.theta += dtheta
        spike = (theta_prev < pi * 0.99 && s.theta >= pi * 0.99) ? 1 : 0
        s.theta = (s.theta + pi) % (2 * pi) - pi
        return spike
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ThetaNeuronState()
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

end # module ThetaAccel
