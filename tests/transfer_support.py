# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_transfer.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, cast
import numpy as np
from numpy.typing import NDArray
import pytest
from sc_neurocore.transfer import (
    TransferConfig,
    SNNCheckpoint,
    apply_transfer_config,
    freeze_layers,
    load_checkpoint,
    save_checkpoint,
    unfreeze_layers,
)
def _make_checkpoint() -> SNNCheckpoint:
    rng = np.random.default_rng(42)
    return SNNCheckpoint(
        weights=[
            rng.normal(size=(32, 64)).astype(np.float64),
            rng.normal(size=(10, 32)).astype(np.float64),
        ],
        layer_names=["hidden", "output"],
        layer_sizes=[(64, 32), (32, 10)],
        neuron_types=["LIF", "LIF"],
        metadata={"task": "mnist", "accuracy": 0.95},
    )
def _write_minimal_checkpoint(
    path: Path,
    weight: NDArray[np.float64] | None = None,
    *,
    layer_sizes: list[list[int]] | None = None,
) -> None:
    archive_weight = np.array([[1.0]], dtype=np.float64) if weight is None else weight
    np.savez_compressed(str(path) + ".npz", layer_0=archive_weight)
    meta = {
        "layer_names": ["hidden"],
        "layer_sizes": [[1, 1]] if layer_sizes is None else layer_sizes,
        "neuron_types": ["LIF"],
        "frozen_layers": [],
        "n_layers": 1,
        "total_params": int(archive_weight.size),
        "metadata": {},
    }
    Path(str(path) + ".json").write_text(json.dumps(meta), encoding="utf-8")

__all__ = ['json', 'Path', 'Sequence', 'cast', 'np', 'NDArray', 'pytest', 'TransferConfig', 'SNNCheckpoint', 'apply_transfer_config', 'freeze_layers', 'load_checkpoint', 'save_checkpoint', 'unfreeze_layers', '_make_checkpoint', '_write_minimal_checkpoint']
