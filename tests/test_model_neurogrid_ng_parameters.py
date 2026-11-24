# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGParameters from former test_model_neurogrid.py

"""Focused suite: TestNGParameters from former test_model_neurogrid.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.neurogrid import (
    NeuroGridNeuron,
)


class TestNGParameters:
    """Parameter behaviour of the NeuroGridNeuron."""

    @pytest.mark.parametrize("g_c", [0.1, 0.5, 1.0])
    def test_coupling_sweep(self, g_c: float) -> None:
        """The model is exercised across a sweep of coupling."""
        n = NeuroGridNeuron(g_c=g_c)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    @pytest.mark.parametrize("delta_t", [1.0, 2.0, 4.0])
    def test_delta_t_sweep(self, delta_t: float) -> None:
        """The model is exercised across a sweep of delta_t."""
        n = NeuroGridNeuron(delta_t=delta_t)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s)

    @pytest.mark.parametrize("tau_d", [20.0, 50.0, 100.0])
    def test_tau_d_sweep(self, tau_d: float) -> None:
        """The model is exercised across a sweep of tau_d."""
        n = NeuroGridNeuron(tau_d=tau_d)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_d)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = NeuroGridNeuron(dt=dt)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)
