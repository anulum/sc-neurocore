# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for transfer/fine_tune

module FineTuneAccel

using Statistics, LinearAlgebra

mutable struct TransferConfigState
    freeze_until::Float64
    lr_backbone::Float64
    lr_head::Float64
end

function TransferConfigState()
    TransferConfigState(-1.0, 0.0, 0.01)
end

function freeze_layers(checkpoint, layer_names, until_index)
    checkpoint: SNNCheckpoint,
    layer_names: list[str] | nothing = nothing,
    until_index: int | nothing = nothing,
    ) -> SNNCheckpoint
    frozen = set(checkpoint.frozen_layers)
    if layer_names is ! nothing
        frozen.update(layer_names)
    if until_index is ! nothing
        for i, name in enumerate(checkpoint.layer_names)
            if i <= until_index
                frozen.add(name)
    checkpoint.frozen_layers = sorted(frozen)
    return checkpoint
end

function unfreeze_layers(checkpoint, layer_names, all_layers)
    checkpoint: SNNCheckpoint,
    layer_names: list[str] | nothing = nothing,
    all_layers: bool = false,
    ) -> SNNCheckpoint
    if all_layers
        checkpoint.frozen_layers = []
        return checkpoint
    if layer_names is ! nothing
        checkpoint.frozen_layers = [n for n in checkpoint.frozen_layers if n ! in layer_names]
    return checkpoint
end

function apply_transfer_config(checkpoint, config)
    checkpoint: SNNCheckpoint,
    config: TransferConfig,
    ) -> tuple[SNNCheckpoint, list[float]]
    if isinstance(config.freeze_until, int) && config.freeze_until >= 0
        freeze_layers(checkpoint, until_index=config.freeze_until)
    elseif isinstance(config.freeze_until, str)
        idx = (
            checkpoint.layer_names.index(config.freeze_until)
            if config.freeze_until in checkpoint.layer_names
            else -1
        )
        if idx >= 0
            freeze_layers(checkpoint, until_index=idx)
    per_layer_lr = []
    for name in checkpoint.layer_names
        if name in checkpoint.frozen_layers
            per_layer_lr = push!(, config.lr_backbone)
        else
            per_layer_lr = push!(, config.lr_head)
    return checkpoint, per_layer_lr
end

end # module FineTuneAccel
