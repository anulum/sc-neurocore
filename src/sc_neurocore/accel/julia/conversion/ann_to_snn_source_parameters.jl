# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia atomic borrowed coefficient replay

"""Borrowed dense coefficients read only during one atomic replay_parameters call.

Inputs/outputs are positive widths; weights are output/input row-major finite
vectors. Bias is absent or one finite per-step bias per output. Threshold is
positive and finite; initial_fraction is finite threshold units. Caller arrays
must stay live and unchanged during snapshot; no borrowed array is returned.
"""
struct LayerParameters
    inputs::Int
    outputs::Int
    weights::Vector{Float64}
    bias::Union{Nothing,Vector{Float64}}
    threshold::Float64
    initial_fraction::Float64
end

"""Atomically snapshot borrowed parameters once and replay the common dense IF computation.

The source vector is a nonempty connected stack. Frames use time/batch/input
ordering and initial states use batch/output ordering per layer. Domain/geometry
refusal raises ArgumentError, numerical overflow OverflowError, budget refusal
OutOfMemoryError. Complete numeric reservation precedes coefficient/frame/state
copies. Caller storage and allocator/interpreter overhead are excluded. All
returned output/state/event vectors are owned independently of caller buffers.
"""
function replay_parameters(source::AbstractVector{LayerParameters}, frames::AbstractVector{<:Real},
                           shape::Tuple{Int,Int}; output_mode::Symbol=:spikes,
                           initial_state::Union{Nothing,AbstractVector}=nothing,
                           trace::Bool=false, binary_inputs::Bool=true,
                           max_working_bytes::Int=256 << 20)
    working_limit(max_working_bytes)
    steps, batch = shape
    steps >= 0 && batch >= 0 && !isempty(source) || throw(ArgumentError("invalid replay geometry"))
    output_mode in (:spikes, :linear) || throw(ArgumentError("invalid output mode"))
    all(l.inputs > 0 && l.outputs > 0 for l in source) || throw(ArgumentError("nonpositive layer width"))
    all(source[i].outputs == source[i+1].inputs for i in 1:length(source)-1) ||
        throw(ArgumentError("disconnected dense stack"))
    BigInt(steps) * batch * source[1].inputs == length(frames) || throw(ArgumentError("invalid frame dimensions"))
    isnothing(initial_state) || length(initial_state) == length(source) ||
        throw(ArgumentError("initial state must have one array per layer"))
    admit_replay(source, steps, batch, trace, output_mode == :linear, max_working_bytes)
    layers = [DenseLayer(l.inputs, l.outputs, l.weights, l.bias, l.threshold;
                        initial_fraction=l.initial_fraction, max_working_bytes=max_working_bytes) for l in source]
    return replay_owned(layers, output_mode, frames, shape; initial_state=initial_state,
                        trace=trace, binary_inputs=binary_inputs)
end
