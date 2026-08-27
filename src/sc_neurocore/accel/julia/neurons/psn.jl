# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of the sliding PSN

"""
k-order sliding Parallel Spiking Neuron — Fang et al. (2023).

`H[t] = sum_{i=0}^{k-1} W_i * X[t-k+1+i]` with zero pre-history,
`S[t] = Theta(H[t] - v_threshold)`, `Theta(0) = 1`, no reset on
firing. The sum accumulates sequentially from i = 0 so the result is
bit-for-bit identical to every other backend.
"""
module PsnAccel

export step!, reset!, simulate, ParallelSpikingNeuronState

mutable struct ParallelSpikingNeuronState
    weights::Vector{Float64}
    history::Vector{Float64}
    v_threshold::Float64
    hidden::Float64
end

function ParallelSpikingNeuronState(kernel_size::Int=8, v_threshold::Float64=1.0)
    kernel_size >= 1 || throw(ArgumentError("kernel_size must be a positive integer"))
    ParallelSpikingNeuronState(
        fill(1.0 / kernel_size, kernel_size),
        zeros(kernel_size),
        v_threshold,
        0.0,
    )
end

function _valid(s::ParallelSpikingNeuronState)
    !isempty(s.weights) &&
        length(s.history) == length(s.weights) &&
        all(isfinite, s.weights) &&
        all(isfinite, s.history) &&
        isfinite(s.v_threshold)
end

function step!(s::ParallelSpikingNeuronState, current::Float64)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    _valid(s) || throw(ArgumentError("sliding PSN state and parameters must be finite"))

    k = length(s.weights)
    hidden = 0.0
    for i in 1:k
        value = i < k ? s.history[i + 1] : current
        hidden += s.weights[i] * value
    end
    isfinite(hidden) || throw(ArgumentError("sliding PSN hidden state became non-finite"))

    for i in 1:(k - 1)
        s.history[i] = s.history[i + 1]
    end
    s.history[k] = current
    s.hidden = hidden
    return hidden >= s.v_threshold ? 1 : 0
end

function reset!(s::ParallelSpikingNeuronState)
    fill!(s.history, 0.0)
    s.hidden = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0)
    s = ParallelSpikingNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.hidden
        spikes += result
    end
    return trace, spikes
end

end # module PsnAccel
