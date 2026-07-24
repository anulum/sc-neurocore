# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_edge.py

from __future__ import annotations

"""Tests for edge power estimation, Sobol generator, and deploy config."""
from sc_neurocore.edge import Board, SobolGenerator
from sc_neurocore.edge.power_estimator import PowerProfile, MemoryFootprint
from sc_neurocore.edge.deploy import generate_cargo_config, generate_memory_x

__all__ = [
    "Board",
    "SobolGenerator",
    "PowerProfile",
    "MemoryFootprint",
    "generate_cargo_config",
    "generate_memory_x",
]
