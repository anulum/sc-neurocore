# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_chiplet_topology.py

from __future__ import annotations

"""Behaviour and validation tests for package topology construction."""
import math
from collections.abc import Callable
import pytest
from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    StackingType,
    TSVLink,
    add_3d_stack,
    make_torus,
    simulate_timing,
)

__all__ = [
    "math",
    "Callable",
    "pytest",
    "ChipletDie",
    "ChipletTopology",
    "InterposerLink",
    "InterposerTech",
    "StackingType",
    "TSVLink",
    "add_3d_stack",
    "make_torus",
    "simulate_timing",
]
