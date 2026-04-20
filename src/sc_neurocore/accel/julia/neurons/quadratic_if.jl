# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for quadratic_if

module QuadraticIfAccel

export step!, simulate, QuadraticIFNeuronState

mutable struct QuadraticIFNeuronState
    v::Float64
    v_reset::Float64
    v_peak::Float64
    dt::Float64
end

function QuadraticIFNeuronState()
    QuadraticIFNeuronState(-1.0, -1.0, 1.0, 0.01)
end

function step!(s::QuadraticIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (s.v ^ 2 + I_ext) * s.dt
        if s.v >= s.v_peak
            s.v = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = QuadraticIFNeuronState()
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

end # module QuadraticIfAccel
