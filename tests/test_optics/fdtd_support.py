# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fdtd.py

from __future__ import annotations

"""Tests for ``FDTDSolver`` (1D) and ``FDTD2DSolver`` (2D Berenger PML).

1D uses a quadratic-ramp absorbing boundary condition (ABC). The 2D
solver implements the split-field Berenger PML with per-direction
conductivities σx/σy and matched-impedance (σ*/μ₀ = σ/ε₀).

These tests exercise:

- Initial-field/grid invariants after construction.
- Injected-pulse energy is non-zero and finite.
- Field values remain bounded (no blow-up) under nominal CFL.
- 1D ABC monotonically damps field amplitude at the grid edges across
  multiple passes.
- 2D Berenger PML ratios σy*(μ₀/ε₀) are set so the matched-impedance
  condition reduces interior reflection.
- Ill-formed material maps are rejected.
"""
import sys
import types
import numpy as np
import pytest
from sc_neurocore.optics.photonic_emitter import (
    FDTD2DSolver,
    FDTDSolver,
    MeepAdapter,
    PhotonicTarget,
)

__all__ = [
    "sys",
    "types",
    "np",
    "pytest",
    "FDTD2DSolver",
    "FDTDSolver",
    "MeepAdapter",
    "PhotonicTarget",
]
