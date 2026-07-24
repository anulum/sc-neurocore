# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_siegert.py

from __future__ import annotations

"""Full pipeline test for SiegertTransferFunction (Siegert 1951).

Mean-field LIF firing rate. Returns float (Hz), NOT int spike.
Analytical: r = [τ_rp + τ_m√π · ∫exp(u²)(1+erf(u))du]⁻¹.
Saturates at 1/τ_rp = 500 Hz. ~524 steps/s (Gauss-Legendre quadrature)."""
import time
import numpy as np
import pytest
from scipy.special import erf as scipy_erf
from sc_neurocore.neurons.models.siegert import SiegertTransferFunction, _erf_approx
from sc_neurocore.network.population import Population

__all__ = [
    "time",
    "np",
    "pytest",
    "scipy_erf",
    "SiegertTransferFunction",
    "_erf_approx",
    "Population",
]
