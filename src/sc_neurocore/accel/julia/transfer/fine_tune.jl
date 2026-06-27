# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia validation mirror for transfer/fine_tune

module FineTuneAccel

export TransferCheckpointState,
    TransferConfigState,
    freeze_layers!,
    unfreeze_layers!,
    apply_transfer_config!,
    validate_fine_tune

mutable struct TransferCheckpointState
    layer_names::Vector{String}
    frozen_layers::Vector{String}

    function TransferCheckpointState(layer_names::Vector{String}, frozen_layers::Vector{String}=String[])
        length(unique(layer_names)) == length(layer_names) ||
            throw(ArgumentError("layer_names must be unique"))
        known = Set(layer_names)
        all(layer -> layer in known, frozen_layers) ||
            throw(ArgumentError("frozen_layers must reference known layers"))
        new(layer_names, sort(unique(frozen_layers)))
    end
end

struct TransferConfigState
    freeze_until::Union{Nothing,Int,String}
    lr_backbone::Float64
    lr_head::Float64

    function TransferConfigState(
        freeze_until::Union{Nothing,Int,String}=nothing,
        lr_backbone::Real=0.0,
        lr_head::Real=0.01,
    )
        backbone = Float64(lr_backbone)
        head = Float64(lr_head)
        isfinite(backbone) && isfinite(head) && backbone >= 0.0 && head >= 0.0 ||
            throw(ArgumentError("learning rates must be finite and non-negative"))
        freeze_until isa Int && freeze_until < 0 &&
            throw(ArgumentError("freeze_until index must be non-negative"))
        new(freeze_until, backbone, head)
    end
end

function freeze_layers!(
    checkpoint::TransferCheckpointState;
    layer_names::Vector{String}=String[],
    until_index::Union{Nothing,Int}=nothing,
)
    _validate_layer_names(checkpoint, layer_names)
    frozen = Set(checkpoint.frozen_layers)
    foreach(layer -> push!(frozen, layer), layer_names)
    if until_index !== nothing
        1 <= until_index <= length(checkpoint.layer_names) ||
            throw(ArgumentError("until_index must reference an existing layer"))
        for name in checkpoint.layer_names[1:until_index]
            push!(frozen, name)
        end
    end
    checkpoint.frozen_layers = sort(collect(frozen))
    checkpoint
end

function unfreeze_layers!(
    checkpoint::TransferCheckpointState;
    layer_names::Vector{String}=String[],
    all_layers::Bool=false,
)
    if all_layers
        checkpoint.frozen_layers = String[]
        return checkpoint
    end
    _validate_layer_names(checkpoint, layer_names)
    removals = Set(layer_names)
    checkpoint.frozen_layers = [name for name in checkpoint.frozen_layers if !(name in removals)]
    checkpoint
end

function apply_transfer_config!(
    checkpoint::TransferCheckpointState,
    config::TransferConfigState,
)
    if config.freeze_until isa Int
        freeze_layers!(checkpoint; until_index=config.freeze_until + 1)
    elseif config.freeze_until isa String
        index = findfirst(==(config.freeze_until), checkpoint.layer_names)
        index === nothing && throw(ArgumentError("freeze_until layer is not present in checkpoint"))
        freeze_layers!(checkpoint; until_index=index)
    end
    [
        name in checkpoint.frozen_layers ? config.lr_backbone : config.lr_head
        for name in checkpoint.layer_names
    ]
end

function validate_fine_tune(config::TransferConfigState)
    isfinite(config.lr_backbone) && isfinite(config.lr_head) &&
        config.lr_backbone >= 0.0 && config.lr_head >= 0.0
end

function _validate_layer_names(checkpoint::TransferCheckpointState, layer_names::Vector{String})
    known = Set(checkpoint.layer_names)
    all(layer -> layer in known, layer_names) || throw(ArgumentError("Unknown layer names"))
    nothing
end

end # module FineTuneAccel
