# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGNumerics from former test_model_marder_stg.py

"""Focused suite: TestSTGNumerics from former test_model_marder_stg.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)


class TestSTGNumerics:
    """Time-step stability, reversal ordering and conductance signs of the MarderSTGNeuron."""

    @pytest.mark.parametrize("dt", [0.025, 0.05])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = MarderSTGNeuron(dt=dt)
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.ca)

    def test_reversal_ordering(self) -> None:
        """The reversal potentials are in physiological order."""
        n = MarderSTGNeuron()
        assert n.e_k < n.e_l < n.e_h < n.e_na

    def test_conductances_non_negative(self) -> None:
        """Conductances non-negative."""
        n = MarderSTGNeuron()
        for g in (n.g_na, n.g_cat, n.g_cas, n.g_a, n.g_kca, n.g_kd, n.g_h, n.g_l):
            assert g >= 0.0
