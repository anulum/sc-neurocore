# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cortical_column_dynamics.py

from __future__ import annotations

"""Tests for CorticalColumn: step, simulate, population outputs, reset, determinism.

Aligned with the production API which uses:
    CorticalColumn(scale=..., seed=...)
    .step(dt=0.1) -> dict[str, np.ndarray]     (one timestep)
    .simulate(duration_ms=..., dt=0.1) -> dict  (full run)
    .reset_state()                               (clear state)

Population keys: L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i.
"""
import numpy as np
from sc_neurocore.network.cortical_column import CorticalColumn

EXPECTED_POPULATIONS = {"L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"}

__all__ = ["np", "CorticalColumn", "EXPECTED_POPULATIONS"]
