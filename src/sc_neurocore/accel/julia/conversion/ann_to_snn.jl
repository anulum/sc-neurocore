# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia deterministic dense IF replay

"""Native dense-if-f64-sequential-v1 replay; source tracing and encoding belong to the frontend."""
module AnnToSnnAccel

export DenseLayer, ConvertedSNN, LayerParameters, ReplayResult, replay, replay_parameters, classify

include("ann_to_snn_resources.jl")
include("ann_to_snn_parameters.jl")

"""Owned output, final states and layer-major complete state/event traces, all row-major."""
struct ReplayResult
    output::Vector{Float64}
    final_state::Vector{Vector{Float64}}
    state_trace::Vector{Vector{Float64}}
    spike_trace::Vector{Vector{Float64}}
end

include("ann_to_snn_compute.jl")
include("ann_to_snn_source_parameters.jl")

"""Replay time/batch/input flat frames with inclusive thresholds and subtractive reset.

Each layer consumes preceding events in the same timestep. Bias follows ordered
separate Float64 multiply/add reductions and is applied every timestep. Explicit
states use batch/output order; traces use time/batch/output order. A linear final
stage returns its cumulative signed integral, while an IF returns incremental
counts. Empty time/batch preserves initial states. All returned storage is owned.
Invalid geometry/domain raises ArgumentError, arithmetic overflow OverflowError,
and an exceeded numeric reservation OutOfMemoryError before frame/state copies.
Caller storage and allocator/interpreter overhead are excluded from the budget.
"""
function replay(model::ConvertedSNN, frames::AbstractVector{<:Real}, shape::Tuple{Int,Int};
                initial_state::Union{Nothing,AbstractVector}=nothing, trace::Bool=false,
                binary_inputs::Bool=true, max_working_bytes::Int=256 << 20)
    steps, batch = shape
    steps >= 0 && batch >= 0 || throw(ArgumentError("negative replay dimensions"))
    isempty(model.layers) && throw(ArgumentError("at least one dense layer required"))
    BigInt(steps) * batch * model.layers[1].inputs == length(frames) ||
        throw(ArgumentError("invalid frame dimensions"))
    isnothing(initial_state) || length(initial_state) == length(model.layers) ||
        throw(ArgumentError("initial state must have one array per layer"))
    admit_replay(model.layers, steps, batch, trace, model.output_mode == :linear, max_working_bytes)
    snapshot = ConvertedSNN(model.layers; output_mode=model.output_mode, max_working_bytes=max_working_bytes)
    return replay_owned(snapshot.layers, snapshot.output_mode, frames, shape;
                        initial_state=initial_state, trace=trace, binary_inputs=binary_inputs)
end

"""Select the first maximal finite output in each row; returns zero-based labels."""
function classify(model::ConvertedSNN, result::ReplayResult, batch::Int)
    width = model.layers[end].outputs
    batch >= 0 && BigInt(batch) * width == length(result.output) && all(isfinite, result.output) ||
        throw(ArgumentError("invalid classification response"))
    labels = Int[]
    for row in 0:batch-1
        best = 0
        for node in 1:width-1
            result.output[row*width+node+1] > result.output[row*width+best+1] && (best = node)
        end
        push!(labels, best)
    end
    return labels
end

end # module AnnToSnnAccel
