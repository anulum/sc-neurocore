# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia validated dense IF computation

"""Replay a previously validated owned layer snapshot; no further coefficient copy.

Callers admit full geometry/storage before invoking this internal computation.
Frames and initial states are copied independently; all ordered float64 dynamics,
complete traces, finite overflow and response semantics are shared by both public
model replay and atomic borrowed-parameter replay.
"""
function replay_owned(layers::Vector{DenseLayer}, output_mode::Symbol,
                      frames::AbstractVector{<:Real}, shape::Tuple{Int,Int};
                      initial_state::Union{Nothing,AbstractVector}=nothing,
                      trace::Bool=false, binary_inputs::Bool=true)
    steps, batch = shape
    owned_frames = Float64.(frames)
    all(v -> isfinite(v) && 0 <= v <= 1 && (!binary_inputs || v == 0 || v == 1), owned_frames) ||
        throw(ArgumentError("input must contain finite bounded currents or binary events"))
    spiking = length(layers) - Int(output_mode == :linear)
    states = Vector{Vector{Float64}}()
    state_trace = Vector{Vector{Float64}}()
    spike_trace = Vector{Vector{Float64}}()
    for (index, layer) in enumerate(layers)
        width = batch * layer.outputs
        if isnothing(initial_state)
            shift = index <= spiking ? layer.initial_fraction * layer.threshold : 0.0
            isfinite(shift) || throw(OverflowError("initial membrane overflow"))
            state = fill(shift, width)
        else
            length(initial_state[index]) == width || throw(ArgumentError("invalid initial state dimensions"))
            state = Float64.(initial_state[index])
            all(isfinite, state) || throw(ArgumentError("initial state must be finite"))
        end
        push!(states, state)
        if trace
            push!(state_trace, zeros(Float64, steps * width))
            index <= spiking && push!(spike_trace, zeros(Float64, steps * width))
        end
    end
    output = zeros(Float64, batch * layers[end].outputs)
    for step in 1:(batch == 0 ? 0 : steps)
        width = batch * layers[1].inputs
        drive = owned_frames[(step-1)*width+1:step*width]
        for (index, layer) in enumerate(layers)
            events = zeros(Float64, batch * layer.outputs)
            for row in 0:batch-1, node in 0:layer.outputs-1
                current = 0.0
                for column in 0:layer.inputs-1
                    product = drive[row*layer.inputs+column+1] * layer.weights[node*layer.inputs+column+1]
                    current = current + product
                end
                isnothing(layer.bias) || (current = current + layer.bias[node+1])
                slot = row * layer.outputs + node + 1
                state = states[index][slot] + current
                isfinite(current) && isfinite(state) || throw(OverflowError("dense replay overflow"))
                if index <= spiking
                    event = Float64(state >= layer.threshold)
                    state = state - event * layer.threshold
                    events[slot] = event
                    trace && (spike_trace[index][(step-1)*batch*layer.outputs+slot] = event)
                    index == length(layers) && (output[slot] = output[slot] + event)
                end
                states[index][slot] = state
                trace && (state_trace[index][(step-1)*batch*layer.outputs+slot] = state)
            end
            drive = events
        end
    end
    output_mode == :linear && (output = copy(states[end]))
    return ReplayResult(output, states, state_trace, spike_trace)
end
