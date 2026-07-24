# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_swarm_control.py

from __future__ import annotations

"""Tests for Neuromorphic Swarm Control (UC4) — 50 tests."""
import unittest
import numpy as np
from sc_neurocore.swarm import (
    SwarmAgent,
    AgentConfig,
    SwarmEnvironment,
    EnvConfig,
    CollectiveFields,
    FieldConfig,
    SwarmFitness,
    SwarmEvolver,
    EvolverConfig,
)

if __name__ == "__main__":
    unittest.main()

__all__ = [
    "unittest",
    "np",
    "SwarmAgent",
    "AgentConfig",
    "SwarmEnvironment",
    "EnvConfig",
    "CollectiveFields",
    "FieldConfig",
    "SwarmFitness",
    "SwarmEvolver",
    "EvolverConfig",
]
