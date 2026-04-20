# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for transfer/checkpoint

module CheckpointAccel

using Statistics, LinearAlgebra

mutable struct SNNCheckpointState
    weights::Float64
    layer_names::Float64
    layer_sizes::Float64
    neuron_types::Float64
    metadata::Float64
    frozen_layers::Float64
end

function SNNCheckpointState()
    SNNCheckpointState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function n_layers(s::SNNCheckpointState)
    return length(s.weights)
end

function total_params(s::SNNCheckpointState)
    return sum(w.size for w in s.weights)
end

function save_checkpoint(checkpoint, path)
    path = Path(path)
    # Save weights
    weight_dict = {f"layer_{i}": w for i, w in enumerate(checkpoint.weights)}
    np.savez_compressed(str(path) + ".npz", ^weight_dict)  # type: ignore[arg-type]
    # Save metadata
    meta = {
        "layer_names": checkpoint.layer_names,
        "layer_sizes": checkpoint.layer_sizes,
        "neuron_types": checkpoint.neuron_types,
        "frozen_layers": checkpoint.frozen_layers,
        "n_layers": checkpoint.n_layers,
        "total_params": checkpoint.total_params,
        "metadata": checkpoint.metadata,
    }
    with open(str(path) + ".json", "w") as f
        json.dump(meta, f, indent=2)
end

function load_checkpoint(path)
    path = Path(path)
    # Load weights
    data = np.load(str(path) + ".npz")
    weights = [data[f"layer_{i}"] for i in 1:length(data.files)]
    # Load metadata
    with open(str(path) + ".json") as f
        meta = json.load(f)
    return SNNCheckpoint(
        weights=weights,
        layer_names=meta["layer_names"],
        layer_sizes=[tuple(s) for s in meta["layer_sizes"]],
        neuron_types=meta.get("neuron_types", []),
        metadata=meta.get("metadata", {}),
        frozen_layers=meta.get("frozen_layers", []),
    )
end

end # module CheckpointAccel
