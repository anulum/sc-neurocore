# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bus_mixed_precision.py

from __future__ import annotations

"""Tests for mixed-precision specifications, solvers, and presets."""
import pytest
from sc_neurocore.compiler.mixed_precision import (
    BlockFloatingPrecisionConfig,
    MixedPrecisionSpec,
    PRECISION_PRESETS,
    PrecisionConfig,
    from_preset,
    solve_precision,
)

__all__ = [
    "pytest",
    "BlockFloatingPrecisionConfig",
    "MixedPrecisionSpec",
    "PRECISION_PRESETS",
    "PrecisionConfig",
    "from_preset",
    "solve_precision",
]
