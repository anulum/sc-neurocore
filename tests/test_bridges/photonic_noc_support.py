# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_photonic_noc.py

from __future__ import annotations

"""Tests for sc_neurocore.bridges.photonic_noc."""
import json
import math
import os
import numpy as np
import pytest
from sc_neurocore.bridges.photonic_noc import (
    CrosstalkAnalyzer,
    MZICompiler,
    MZIGate,
    PhotonicCircuitDesign,
    PowerBudgetAnalyzer,
    SCToPhotonic,
    ThermalPhaseShifter,
    WDMAssigner,
    WDMChannel,
    WaveguideRouter,
    WaveguideSegment,
    WaveguideType,
    export_photonic_json,
    visualize_photonic,
)
@pytest.fixture
def simple_adjacency() -> np.ndarray:
    """4-node mesh network."""
    return np.array(
        [
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.8],
            [0.5, 0.0, 0.0, 1.0],
            [0.0, 0.8, 1.0, 0.0],
        ]
    )
@pytest.fixture
def simple_design(simple_adjacency: np.ndarray) -> PhotonicCircuitDesign:
    """Compiled 4-node photonic design."""
    compiler = SCToPhotonic()
    return compiler.compile(simple_adjacency, name="test_noc")

__all__ = ['json', 'math', 'os', 'np', 'pytest', 'CrosstalkAnalyzer', 'MZICompiler', 'MZIGate', 'PhotonicCircuitDesign', 'PowerBudgetAnalyzer', 'SCToPhotonic', 'ThermalPhaseShifter', 'WDMAssigner', 'WDMChannel', 'WaveguideRouter', 'WaveguideSegment', 'WaveguideType', 'export_photonic_json', 'visualize_photonic', 'simple_adjacency', 'simple_design']
