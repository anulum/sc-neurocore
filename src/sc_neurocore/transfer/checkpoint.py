# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN checkpoint serialization

"""Validated checkpoint serialization for transfer-learning SNN state.

The transfer checkpoint format stores dense layer weights in a compressed
``.npz`` archive and the structural contract in a sidecar JSON document.
Loading always disables pickle and reconstructs the same validation path used
by in-memory checkpoint construction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class _Metadata(TypedDict):
    """Validated metadata sidecar fields used to rebuild a checkpoint."""

    layer_names: list[str]
    layer_sizes: list[tuple[int, int]]
    neuron_types: list[str]
    frozen_layers: list[str]
    n_layers: int
    total_params: int
    metadata: dict[str, object]


@dataclass
class SNNCheckpoint:
    """Complete dense-weight SNN checkpoint for transfer workflows.

    Parameters
    ----------
    weights:
        One two-dimensional weight matrix per layer. A matrix shape must be
        ``(output_features, input_features)`` for the matching ``layer_sizes``
        entry ``(input_features, output_features)``.
    layer_names:
        Unique layer names in forward order.
    layer_sizes:
        ``(input_features, output_features)`` pairs for each layer.
    neuron_types:
        Optional neuron-model labels, either empty or one label per layer.
    metadata:
        JSON-serializable provenance or training metadata.
    frozen_layers:
        Layer names currently marked non-trainable.
    """

    weights: list[FloatArray]
    layer_names: list[str]
    layer_sizes: list[tuple[int, int]]
    neuron_types: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    frozen_layers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize arrays and reject inconsistent checkpoint state."""
        _validate_string_vector(self.layer_names, "layer_names")
        if len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("Checkpoint layer_names must be unique")
        if len(self.weights) != len(self.layer_names):
            raise ValueError("Checkpoint weights length must match layer_names")
        if len(self.layer_sizes) != len(self.layer_names):
            raise ValueError("Checkpoint layer_sizes length must match layer_names")
        self.layer_sizes = [_validate_layer_size_tuple(size) for size in self.layer_sizes]
        if self.neuron_types:
            _validate_string_vector(self.neuron_types, "neuron_types")
            if len(self.neuron_types) != len(self.layer_names):
                raise ValueError("Checkpoint neuron_types length must match layer_names")
        _validate_string_vector(self.frozen_layers, "frozen_layers")
        unknown_frozen = sorted(set(self.frozen_layers) - set(self.layer_names))
        if unknown_frozen:
            raise ValueError("Checkpoint frozen_layers must reference known layers")
        self.frozen_layers = sorted(set(self.frozen_layers))
        _validate_json_serializable(self.metadata)
        self.weights = [
            _validate_weight_array(weight, f"layer_{index}", self.layer_sizes[index])
            for index, weight in enumerate(self.weights)
        ]

    @property
    def n_layers(self) -> int:
        """Return the number of serialized layers."""
        return len(self.weights)

    @property
    def total_params(self) -> int:
        """Return the total number of scalar weight parameters."""
        return int(sum(weight.size for weight in self.weights))


def save_checkpoint(checkpoint: SNNCheckpoint, path: str | Path) -> None:
    """Save an SNN checkpoint to ``path.npz`` plus ``path.json``.

    Parameters
    ----------
    checkpoint:
        Validated checkpoint to serialize.
    path:
        Base path without extension. Parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    weight_dict = {f"layer_{index}": weight for index, weight in enumerate(checkpoint.weights)}
    np.savez_compressed(_npz_path(path), **weight_dict)  # type: ignore[arg-type]

    meta = {
        "layer_names": checkpoint.layer_names,
        "layer_sizes": checkpoint.layer_sizes,
        "neuron_types": checkpoint.neuron_types,
        "frozen_layers": checkpoint.frozen_layers,
        "n_layers": checkpoint.n_layers,
        "total_params": checkpoint.total_params,
        "metadata": checkpoint.metadata,
    }
    _json_path(path).write_text(
        json.dumps(meta, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(path: str | Path) -> SNNCheckpoint:
    """Load and validate an SNN checkpoint from ``path.npz`` and ``path.json``.

    Parameters
    ----------
    path:
        Base path without extension.

    Returns
    -------
    SNNCheckpoint:
        Reconstructed checkpoint with finite ``float64`` weight arrays.
    """
    path = Path(path)

    with _json_path(path).open(encoding="utf-8") as handle:
        raw_meta: object = json.load(handle)
    meta = _validate_metadata(raw_meta)
    n_layers = meta["n_layers"]

    with np.load(_npz_path(path), allow_pickle=False) as data:
        expected_keys = [f"layer_{index}" for index in range(n_layers)]
        if set(data.files) != set(expected_keys):
            raise ValueError("Checkpoint weight archive does not match metadata layers")
        weights = [
            _validate_weight_array(data[key], key, meta["layer_sizes"][index])
            for index, key in enumerate(expected_keys)
        ]

    checkpoint = SNNCheckpoint(
        weights=weights,
        layer_names=meta["layer_names"],
        layer_sizes=meta["layer_sizes"],
        neuron_types=meta["neuron_types"],
        metadata=meta["metadata"],
        frozen_layers=meta["frozen_layers"],
    )
    if checkpoint.total_params != meta["total_params"]:
        raise ValueError("Checkpoint metadata total_params does not match weights")
    return checkpoint


def _validate_metadata(meta: object) -> _Metadata:
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
    layer_sizes = [_validate_layer_size_list(size) for size in layer_sizes_raw]
    if len(layer_sizes) != n_layers:
        raise ValueError("Checkpoint metadata layer_sizes length does not match n_layers")

    neuron_types = _optional_string_list(meta, "neuron_types")
    if neuron_types and len(neuron_types) != n_layers:
        raise ValueError("Checkpoint metadata neuron_types length does not match n_layers")

    frozen_layers = _optional_string_list(meta, "frozen_layers")
    metadata = meta.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata field must be a JSON object")
    normalized_metadata = dict(metadata)
    _validate_json_serializable(normalized_metadata)

    total_params = meta.get("total_params", 0)
    if not isinstance(total_params, int) or isinstance(total_params, bool) or total_params < 0:
        raise ValueError("Checkpoint metadata total_params must be a non-negative integer")

    return {
        "layer_names": layer_names,
        "layer_sizes": layer_sizes,
        "neuron_types": neuron_types,
        "frozen_layers": frozen_layers,
        "n_layers": n_layers,
        "total_params": total_params,
        "metadata": normalized_metadata,
    }


def _require_string_list(meta: Mapping[str, object], key: str) -> list[str]:
    value = meta.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint metadata {key} must be a list of strings")
    return value


def _optional_string_list(meta: Mapping[str, object], key: str) -> list[str]:
    value = meta.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint metadata {key} must be a list of strings")
    return value


def _validate_layer_size_list(size: object) -> tuple[int, int]:
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in size)
    ):
        raise ValueError("Checkpoint metadata layer_sizes entries must be integer pairs")
    if size[0] < 0 or size[1] < 0:
        raise ValueError("Checkpoint metadata layer_sizes entries must be non-negative")
    return (size[0], size[1])


def _validate_layer_size_tuple(size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in size)
    ):
        raise ValueError("Checkpoint layer_sizes entries must be integer pairs")
    if size[0] < 0 or size[1] < 0:
        raise ValueError("Checkpoint layer_sizes entries must be non-negative")
    return size


def _validate_weight_array(
    array: NDArray[Any],
    key: str,
    layer_size: tuple[int, int],
) -> FloatArray:
    if array.dtype.hasobject:
        raise ValueError(f"Checkpoint weight {key} must not contain Python objects")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Checkpoint weight {key} must be numeric")
    normalized = np.asarray(array, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError(f"Checkpoint weight {key} must be a two-dimensional array")
    expected_shape = (layer_size[1], layer_size[0])
    if normalized.shape != expected_shape:
        raise ValueError(
            f"Checkpoint weight {key} shape must match layer_sizes as {expected_shape}"
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"Checkpoint weight {key} must contain finite numeric values")
    return normalized


def _validate_string_vector(values: list[str], label: str) -> None:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"Checkpoint {label} must be a list of strings")


def _validate_json_serializable(value: Mapping[str, object]) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint metadata must be JSON serializable") from exc


def _npz_path(path: Path) -> str:
    """Return the weight archive path for a checkpoint base path."""
    return str(path) + ".npz"


def _json_path(path: Path) -> Path:
    """Return the metadata sidecar path for a checkpoint base path."""
    return Path(str(path) + ".json")
