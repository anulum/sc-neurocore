# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN checkpoint serialization

"""Save and load complete SNN model state: weights, architecture, metadata.

No SNN framework has proper checkpoint serialization that preserves
all state needed to resume training or deploy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SNNCheckpoint:
    """Complete SNN model checkpoint."""

    weights: list[np.ndarray[Any, Any]]
    layer_names: list[str]
    layer_sizes: list[tuple[int, int]]
    neuron_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    frozen_layers: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def total_params(self) -> int:
        return sum(w.size for w in self.weights)


def save_checkpoint(checkpoint: SNNCheckpoint, path: str | Path) -> None:
    """Save SNN checkpoint to .npz + .json.

    Parameters
    ----------
    checkpoint : SNNCheckpoint
    path : str or Path
        Base path (without extension). Creates path.npz and path.json.
    """
    path = Path(path)

    # Save weights
    weight_dict = {f"layer_{i}": w for i, w in enumerate(checkpoint.weights)}
    np.savez_compressed(str(path) + ".npz", **weight_dict)  # type: ignore[arg-type]

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
    with open(str(path) + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str | Path) -> SNNCheckpoint:
    """Load SNN checkpoint from .npz + .json.

    Parameters
    ----------
    path : str or Path
        Base path (without extension).

    Returns
    -------
    SNNCheckpoint
    """
    path = Path(path)

    with open(str(path) + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    meta = _validate_metadata(meta)
    n_layers = meta["n_layers"]

    with np.load(str(path) + ".npz", allow_pickle=False) as data:
        expected_keys = [f"layer_{i}" for i in range(n_layers)]
        if set(data.files) != set(expected_keys):
            raise ValueError("Checkpoint weight archive does not match metadata layers")
        weights = [_validate_weight_array(data[key], key) for key in expected_keys]

    return SNNCheckpoint(
        weights=weights,
        layer_names=meta["layer_names"],
        layer_sizes=[tuple(s) for s in meta["layer_sizes"]],
        neuron_types=meta.get("neuron_types", []),
        metadata=meta.get("metadata", {}),
        frozen_layers=meta.get("frozen_layers", []),
    )


def _validate_metadata(meta: object) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ValueError("Checkpoint metadata must be a JSON object")

    n_layers = meta.get("n_layers")
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 0:
        raise ValueError("Checkpoint metadata n_layers must be a non-negative integer")

    layer_names = _require_string_list(meta, "layer_names")
    if len(layer_names) != n_layers:
        raise ValueError("Checkpoint metadata layer_names length does not match n_layers")

    layer_sizes_raw = meta.get("layer_sizes")
    if not isinstance(layer_sizes_raw, list):
        raise ValueError("Checkpoint metadata layer_sizes must be a list")
    layer_sizes = [_validate_layer_size(size) for size in layer_sizes_raw]
    if len(layer_sizes) != n_layers:
        raise ValueError("Checkpoint metadata layer_sizes length does not match n_layers")

    neuron_types = _optional_string_list(meta, "neuron_types")
    if neuron_types and len(neuron_types) != n_layers:
        raise ValueError("Checkpoint metadata neuron_types length does not match n_layers")

    frozen_layers = _optional_string_list(meta, "frozen_layers")
    metadata = meta.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata field must be a JSON object")

    return {
        "layer_names": layer_names,
        "layer_sizes": layer_sizes,
        "neuron_types": neuron_types,
        "frozen_layers": frozen_layers,
        "n_layers": n_layers,
        "total_params": meta.get("total_params", 0),
        "metadata": dict(metadata),
    }


def _require_string_list(meta: Mapping[str, Any], key: str) -> list[str]:
    value = meta.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint metadata {key} must be a list of strings")
    return value


def _optional_string_list(meta: Mapping[str, Any], key: str) -> list[str]:
    value = meta.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint metadata {key} must be a list of strings")
    return value


def _validate_layer_size(size: object) -> list[int]:
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in size)
    ):
        raise ValueError("Checkpoint metadata layer_sizes entries must be integer pairs")
    if size[0] < 0 or size[1] < 0:
        raise ValueError("Checkpoint metadata layer_sizes entries must be non-negative")
    return size


def _validate_weight_array(array: np.ndarray[Any, Any], key: str) -> np.ndarray[Any, Any]:
    if array.dtype.hasobject:
        raise ValueError(f"Checkpoint weight {key} must not contain Python objects")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Checkpoint weight {key} must be numeric")
    return np.asarray(array)
