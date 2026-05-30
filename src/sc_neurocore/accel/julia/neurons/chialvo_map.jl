# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for chialvo_map

module ChialvoMapAccel

export step!, simulate, validate, ChialvoMapNeuronState

mutable struct ChialvoMapNeuronState
    x::Float64
    y::Float64
    a::Float64
    b::Float64
    c::Float64
    k::Float64
    x_threshold::Float64
end

function ChialvoMapNeuronState()
    ChialvoMapNeuronState(0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0)
end

function validate(s::ChialvoMapNeuronState)::Bool
    return isfinite(s.x) &&
        isfinite(s.y) &&
        isfinite(s.a) &&
        isfinite(s.b) &&
        isfinite(s.c) &&
        isfinite(s.k) &&
        isfinite(s.x_threshold)
end

safe_exp(value::Float64)::Float64 = exp(clamp(value, -745.0, 709.0))

function step!(s::ChialvoMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end

    x_prev = s.x
    x_new = s.x ^ 2 * safe_exp(s.y - s.x) + s.k + I_ext
    y_new = s.a * s.y - s.b * s.x + s.c
    if !isfinite(x_new) || !isfinite(y_new)
        return -1
    end
    s.x = x_new
    s.y = y_new
    return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ChialvoMapNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.x
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ChialvoMapAccel
