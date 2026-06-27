# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia validation mirror for transfer/checkpoint

module CheckpointAccel

export SNNCheckpointState, n_layers, total_params, validate_checkpoint

mutable struct SNNCheckpointState
    weights::Vector{Matrix{Float64}}
    layer_names::Vector{String}
    layer_sizes::Vector{Tuple{Int,Int}}
    neuron_types::Vector{String}
    frozen_layers::Vector{String}

    function SNNCheckpointState(
        weights::Vector{Matrix{Float64}},
        layer_names::Vector{String},
        layer_sizes::Vector{Tuple{Int,Int}},
        neuron_types::Vector{String}=String[],
        frozen_layers::Vector{String}=String[],
    )
        state = new(weights, layer_names, layer_sizes, neuron_types, sort(unique(frozen_layers)))
        _validate_checkpoint!(state)
        state
    end
end

function n_layers(state::SNNCheckpointState)
    length(state.weights)
end

function total_params(state::SNNCheckpointState)
    sum(length(weight) for weight in state.weights)
end

function validate_checkpoint(state::SNNCheckpointState)
    try
        _validate_checkpoint!(state)
        true
    catch
        false
    end
end

function _validate_checkpoint!(state::SNNCheckpointState)
    length(state.weights) == length(state.layer_names) ||
        throw(ArgumentError("weights length must match layer_names"))
    length(state.layer_sizes) == length(state.layer_names) ||
        throw(ArgumentError("layer_sizes length must match layer_names"))
    length(unique(state.layer_names)) == length(state.layer_names) ||
        throw(ArgumentError("layer_names must be unique"))
    (isempty(state.neuron_types) || length(state.neuron_types) == length(state.layer_names)) ||
        throw(ArgumentError("neuron_types length must match layer_names"))

    known = Set(state.layer_names)
    all(layer -> layer in known, state.frozen_layers) ||
        throw(ArgumentError("frozen_layers must reference known layers"))

    for (index, weight) in enumerate(state.weights)
        inputs, outputs = state.layer_sizes[index]
        size(weight) == (outputs, inputs) ||
            throw(ArgumentError("layer_$(index - 1) shape must match layer_sizes"))
        all(isfinite, weight) ||
            throw(ArgumentError("layer_$(index - 1) weights must be finite"))
    end
    nothing
end

end # module CheckpointAccel
