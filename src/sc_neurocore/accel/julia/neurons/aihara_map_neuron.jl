# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for aihara_map_neuron

module AiharaMapNeuronAccel

export step!, simulate, validate, AiharaMapNeuronState

mutable struct AiharaMapNeuronState
    x::Float64
    y::Float64
    k_f::Float64
    k_s::Float64
    alpha::Float64
    delta::Float64
    x_threshold::Float64
end

function AiharaMapNeuronState()
    AiharaMapNeuronState(0.0, 0.0, 0.7, 0.95, 2.0, 0.05, 0.5)
end

function validate(s::AiharaMapNeuronState)::Bool
    return isfinite(s.x) &&
        isfinite(s.y) &&
        isfinite(s.k_f) &&
        s.k_f >= 0.0 &&
        isfinite(s.k_s) &&
        isfinite(s.alpha) &&
        isfinite(s.delta) &&
        s.delta >= 0.0 &&
        isfinite(s.x_threshold)
end

function logistic(z::Float64)::Float64
    if z >= 0.0
        return 1.0 / (1.0 + exp(-z))
    end
    exp_z = exp(z)
    return exp_z / (1.0 + exp_z)
end

function step!(s::AiharaMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end

    x_prev = s.x
    sigmoid = logistic(s.x + s.alpha)
    x_new = s.k_f * s.x * sigmoid - s.y + I_ext
    y_new = s.k_s * s.y + s.delta * s.x
    if !isfinite(x_new) || !isfinite(y_new)
        return -1
    end
    s.x = max(-10.0, min(10.0, x_new))
    s.y = max(-10.0, min(10.0, y_new))
    return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AiharaMapNeuronState()
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

end # module AiharaMapNeuronAccel
