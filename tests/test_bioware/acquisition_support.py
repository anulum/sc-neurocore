# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_acquisition.py

from __future__ import annotations

"""Tests for MEA acquisition, spike detection, sorting, and artifacts."""
import sys
from collections.abc import MutableMapping
from typing import cast
import numpy as np
import numpy.typing as npt
import pytest
from sc_neurocore.bioware.bioware import (
    ArtifactRejector,
    DetectedSpike,
    MEAConfig,
    SpikeDetector,
    SpikeSorter,
)

FloatArray = npt.NDArray[np.float64]


def _synth_voltage(
    n_samples: int = 1000,
    n_channels: int = 10,
    seed: int = 42,
) -> FloatArray:
    """Generate synthetic MEA voltage data with embedded spikes."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 5, size=(n_samples, n_channels))
    for i in range(0, n_samples, 200):
        data[i, 0] = -80.0
        if i + 50 < n_samples:
            data[i + 50, 3] = -60.0
    return data


__all__ = [
    "sys",
    "MutableMapping",
    "cast",
    "np",
    "npt",
    "pytest",
    "ArtifactRejector",
    "DetectedSpike",
    "MEAConfig",
    "SpikeDetector",
    "SpikeSorter",
    "FloatArray",
    "_synth_voltage",
]
