# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_analysis.py

from __future__ import annotations

"""Tests for culture, LFP, burst, and latency analysis."""
from typing import Any
import numpy as np
import pytest
from sc_neurocore.bioware.bioware import (
    CultureHealth,
    DetectedSpike,
    LatencyBudget,
    LFPBand,
    detect_network_bursts,
    extract_lfp_power,
)


def _synth_voltage(
    n_samples: int,
    n_channels: int,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
    """Generate deterministic finite voltage data for spectral tests."""
    return np.random.default_rng(seed).normal(0.0, 5.0, size=(n_samples, n_channels))


__all__ = [
    "Any",
    "np",
    "pytest",
    "CultureHealth",
    "DetectedSpike",
    "LatencyBudget",
    "LFPBand",
    "detect_network_bursts",
    "extract_lfp_power",
    "_synth_voltage",
]
