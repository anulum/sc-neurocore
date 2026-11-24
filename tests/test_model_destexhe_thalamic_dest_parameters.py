# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDestParameters from former test_model_destexhe_thalamic.py

"""Focused suite: TestDestParameters from former test_model_destexhe_thalamic.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.destexhe_thalamic import (
    DestexheThalamicNeuron,
)


class TestDestParameters:
    """Parameter behaviour of the DestexheThalamicNeuron."""

    @pytest.mark.parametrize("g_t", [0.0, 2.0, 5.0])
    def test_g_t_sweep(self, g_t: float) -> None:
        """The state stays finite across T-current conductances."""
        n = DestexheThalamicNeuron(g_t=g_t)
        for _ in range(3000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_na", [50.0, 100.0, 150.0])
    def test_g_na_sweep(self, g_na: float) -> None:
        """The state stays finite across sodium conductances."""
        n = DestexheThalamicNeuron(g_na=g_na)
        for _ in range(3000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = DestexheThalamicNeuron(dt=dt)
        for _ in range(3000):
            n.step(5.0)
        assert np.isfinite(n.v)
