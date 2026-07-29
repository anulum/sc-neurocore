# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for the SC two-state chaotic map

module SCChaoticMapNeuronAccel

export step!, simulate, simulate_sc_chaotic_map, validate, SCChaoticMapNeuronState

mutable struct SCChaoticMapNeuronState
    x::Float64
    y::Float64
    k_f::Float64
    k_s::Float64
    alpha::Float64
    delta::Float64
    x_threshold::Float64
end

function SCChaoticMapNeuronState()
    SCChaoticMapNeuronState(0.0, 0.0, 0.7, 0.95, 2.0, 0.05, 0.5)
end

function validate(s::SCChaoticMapNeuronState)::Bool
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

function step!(s::SCChaoticMapNeuronState, current::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(current) || !isfinite(dt)
        return -1
    end

    x_prev = s.x
    x_new = s.k_f * s.x * logistic(s.x + s.alpha) - s.y + current
    y_new = s.k_s * s.y + s.delta * s.x
    if !isfinite(x_new) || !isfinite(y_new)
        return -1
    end
    s.x = clamp(x_new, -10.0, 10.0)
    s.y = clamp(y_new, -10.0, 10.0)
    return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; current::Float64=10.0, dt::Float64=0.1)
    s = SCChaoticMapNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, current; dt=dt)
        trace[t] = s.x
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

function simulate_sc_chaotic_map(
    x::Real,
    y::Real,
    k_f::Real,
    k_s::Real,
    alpha::Real,
    delta::Real,
    x_threshold::Real,
    current::AbstractVector,
)
    state = SCChaoticMapNeuronState(Float64.((x, y, k_f, k_s, alpha, delta, x_threshold))...)
    validate(state) || throw(ArgumentError("invalid SC chaotic-map configuration"))
    all(isfinite, current) || throw(ArgumentError("current must contain only finite values"))
    steps = length(current)
    x_trace = Vector{Float64}(undef, steps)
    y_trace = Vector{Float64}(undef, steps)
    spikes = Vector{Float64}(undef, steps)
    count = 0
    @inbounds for index in 1:steps
        event = step!(state, Float64(current[index]))
        event >= 0 || throw(ArgumentError("invalid SC chaotic-map candidate"))
        x_trace[index] = state.x
        y_trace[index] = state.y
        spikes[index] = Float64(event)
        count += event
    end
    return (x_trace, y_trace, spikes, state.x, state.y, count)
end

end # module SCChaoticMapNeuronAccel
