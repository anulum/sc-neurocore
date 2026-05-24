# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fractional_lif

module FractionalLifAccel

export step!, simulate, valid, FractionalLIFNeuronState

mutable struct FractionalLIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    alpha::Float64
    resistance::Float64
    dt::Float64
    max_history::Int
    history::Vector{Float64}
    gl_coeffs::Vector{Float64}
end

function FractionalLIFNeuronState()
    max_history = 100
    FractionalLIFNeuronState(
        0.0,
        0.0,
        0.0,
        1.0,
        0.8,
        1.0,
        1.0,
        max_history,
        zeros(max_history),
        compute_gl_coefficients(0.8, max_history),
    )
end

function compute_gl_coefficients(alpha::Float64, max_history::Int)
    coeffs = ones(Float64, max_history)
    for k in 2:max_history
        coeffs[k] = coeffs[k - 1] * ((k - 1) - 1 - alpha) / (k - 1)
    end
    return coeffs
end

function step!(s::FractionalLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !valid(s) || !isfinite(I_ext)
        return 0
    end
    rhs = -(s.v - s.v_rest) + s.resistance * I_ext
    terms = min(length(s.history), s.max_history, length(s.gl_coeffs))
    gl_sum = 0.0
    for k in 2:terms
        gl_sum += s.gl_coeffs[k] * s.history[end - (k - 2)]
    end
    s.v = rhs * s.dt ^ s.alpha - gl_sum
    push!(s.history, s.v)
    if length(s.history) > s.max_history
        popfirst!(s.history)
    end
    if s.v >= s.v_threshold
        s.v = s.v_reset
        s.history[end] = s.v_reset
        return 1
    end
    return 0
end

function valid(s::FractionalLIFNeuronState)
    return isfinite(s.v) &&
        isfinite(s.v_rest) &&
        isfinite(s.v_reset) &&
        isfinite(s.v_threshold) &&
        isfinite(s.alpha) &&
        s.alpha > 0.0 &&
        s.alpha <= 1.0 &&
        isfinite(s.resistance) &&
        s.resistance > 0.0 &&
        isfinite(s.dt) &&
        s.dt > 0.0 &&
        s.max_history > 1 &&
        length(s.history) == s.max_history &&
        length(s.gl_coeffs) == s.max_history &&
        all(isfinite, s.history) &&
        all(isfinite, s.gl_coeffs)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FractionalLIFNeuronState()
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

end # module FractionalLifAccel
