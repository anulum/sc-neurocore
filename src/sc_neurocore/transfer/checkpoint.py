# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SNNCheckpoint:
    """Complete SNN model checkpoint."""

    weights: list[np.ndarray]
    layer_names: list[str]
    layer_sizes: list[tuple[int, int]]
    neuron_types: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    frozen_layers: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def total_params(self) -> int:
        return sum(w.size for w in self.weights)


def save_checkpoint(checkpoint: SNNCheckpoint, path: str | Path):
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
    np.savez_compressed(str(path) + ".npz", **weight_dict)

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

    # Load weights
    data = np.load(str(path) + ".npz")
    weights = [data[f"layer_{i}"] for i in range(len(data.files))]

    # Load metadata
    with open(str(path) + ".json") as f:
        meta = json.load(f)

    return SNNCheckpoint(
        weights=weights,
        layer_names=meta["layer_names"],
        layer_sizes=[tuple(s) for s in meta["layer_sizes"]],
        neuron_types=meta.get("neuron_types", []),
        metadata=meta.get("metadata", {}),
        frozen_layers=meta.get("frozen_layers", []),
    )
