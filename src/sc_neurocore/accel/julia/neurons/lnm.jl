# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for lnm

module LnmAccel

export step!, simulate, LearnableNeuronModelState

mutable struct LearnableNeuronModelState
    v::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    v_threshold::Float64
    v_reset::Float64
    f_slope::Float64
    f_shift::Float64
end

function LearnableNeuronModelState()
    LearnableNeuronModelState(0.0, 0.9, 0.1, 0.05, 1.0, 0.0, 5.0, 0.5)
end

function validate(s::LearnableNeuronModelState)::Bool
    isfinite(s.v) &&
    isfinite(s.alpha) &&
    isfinite(s.beta) &&
    isfinite(s.gamma) &&
    isfinite(s.v_threshold) && s.v_threshold > 0.0 &&
    isfinite(s.v_reset) &&
    isfinite(s.f_slope) && s.f_slope > 0.0 &&
    isfinite(s.f_shift)
end

function _sigmoid(value::Float64)::Float64
    if value >= 0.0
        z = exp(-value)
        return 1.0 / (1.0 + z)
    end
    z = exp(value)
    return z / (1.0 + z)
end

function step!(s::LearnableNeuronModelState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return 0
    end

    f_v = _sigmoid(s.f_slope * (s.v - s.f_shift))
    s.v = s.alpha * s.v + s.beta * I_ext + s.gamma * f_v
    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LearnableNeuronModelState()
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

end # module LnmAccel
