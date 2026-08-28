# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_glif.py

from __future__ import annotations

"""GLIF schema, hand-model, and Q16.16 RTL parity contracts."""
from pathlib import Path
from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
import pytest
from tests.cosim_support import (
    HAS_IVERILOG,
    _glif_hand_spike_count,
    _python_spike_count,
    _verilog_compiles,
    _verilog_spike_count_q1616,
)

_TRANSCENDENTAL_COMPILE_MODELS = ["sc_four_state_glif"]

__all__ = [
    "Path",
    "GLIFNeuron",
    "SCFourStateGLIFNeuron",
    "UniversalNeuron",
    "pytest",
    "HAS_IVERILOG",
    "_glif_hand_spike_count",
    "_python_spike_count",
    "_verilog_compiles",
    "_verilog_spike_count_q1616",
    "_TRANSCENDENTAL_COMPILE_MODELS",
]
