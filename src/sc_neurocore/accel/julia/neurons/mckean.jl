# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mckean

module MckeanAccel

export step!, simulate, McKeanNeuronState

mutable struct McKeanNeuronState
    v::Float64
    w::Float64
    a::Float64
    epsilon::Float64
    gamma::Float64
    dt::Float64
    v_peak::Float64
end

function McKeanNeuronState()
    McKeanNeuronState(0.0, 0.0, 0.25, 0.01, 0.5, 0.1, 0.8)
end

function _f(s::McKeanNeuronState, v)
    mid1 = s.a / 2.0
    mid2 = (1.0 + s.a) / 2.0
    if v < mid1
        return -v
    elseif v < mid2
        return v - s.a
    else
        return 1.0 - v
    end
end

function step!(s::McKeanNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dv = (s._f(s.v) - s.w + I_ext) * s.dt
        dw = s.epsilon * (s.v - s.gamma * s.w) * s.dt
        v_prev = s.v
        s.v += dv
        s.w += dw
        return (s.v >= s.v_peak && v_prev < s.v_peak) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = McKeanNeuronState()
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

end # module MckeanAccel
