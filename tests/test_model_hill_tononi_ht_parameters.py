# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHTParameters from former test_model_hill_tononi.py

"""Focused suite: TestHTParameters from former test_model_hill_tononi.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.sc_six_state_thalamocortical import (
    SCSixStateThalamocorticalNeuron,
)


class TestHTParameters:
    """Parameter behaviour of the SCSixStateThalamocorticalNeuron."""

    @pytest.mark.parametrize("g_kna", [0.0, 1.33, 3.0])
    def test_g_kna_sweep(self, g_kna: float) -> None:
        """The model is exercised across a sweep of g_kna."""
        n = SCSixStateThalamocorticalNeuron(g_kna=g_kna)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_t", [0.0, 3.0, 6.0])
    def test_g_t_sweep(self, g_t: float) -> None:
        """The model is exercised across a sweep of g_t."""
        n = SCSixStateThalamocorticalNeuron(g_t=g_t)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = SCSixStateThalamocorticalNeuron(dt=dt)
        for _ in range(10_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.na_i)
