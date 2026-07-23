# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_wilson_cowan_dynamics.py

from __future__ import annotations

"""Multi-angle tests checking published dynamical properties of the
Wilson-Cowan 1972 E/I rate model, not just API / parity.

Sections:
  1. Sigmoid transfer function — regime coverage
  2. Fixed point at zero drive — quiescent attractor is stable
  3. Monotone response — stronger drive → higher E
  4. E/I separation — E responds faster than I (τ_e < τ_i)
  5. Limit-cycle behaviour in the oscillator regime
  6. Bounded state — 0 ≤ E, I ≤ 1 for physically reasonable
     parameter grid
  7. Parameter sweeps — asymmetric E/I coupling produces expected
     phase-space structure
  8. Cross-backend parity under extreme parameter regimes
  9. Edge cases — zero-length, single-step, boundary init
"""
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit
DEFAULT_PARAMS = dict(
    w_ee=10.0,
    w_ei=6.0,
    w_ie=10.0,
    w_ii=1.0,
    tau_e=1.0,
    tau_i=2.0,
    a=1.2,
    theta=4.0,
    dt=0.1,
)

__all__ = ['math', 'np', 'pytest', 'WilsonCowanUnit', 'DEFAULT_PARAMS', '__all__']
