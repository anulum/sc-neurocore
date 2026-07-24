# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_pernarowski.py

from __future__ import annotations

"""Pernarowski schema, hand-model, and Q16.16 RTL parity contracts."""
from pathlib import Path
import pytest
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _pernarowski_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)

__all__ = [
    "Path",
    "pytest",
    "PernarowskiNeuron",
    "UniversalNeuron",
    "HAS_IVERILOG",
    "_pernarowski_hand_spike_count",
    "_python_spike_count",
    "_verilog_spike_count_q1616",
]
