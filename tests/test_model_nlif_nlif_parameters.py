# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFParameters from former test_model_nlif.py

"""Focused suite: TestNLIFParameters from former test_model_nlif.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.nlif import (
    NonlinearLIFNeuron,
)
from tests.model_nlif_support import (
    _run,
)


class TestNLIFParameters:
    """Parameter behaviour of the NonlinearLIFNeuron."""

    @pytest.mark.parametrize("a", [0.02, 0.04, 0.08])
    def test_a_nonlinearity(self, a: float) -> None:
        """The voltage stays finite across the nonlinearity a."""
        n = NonlinearLIFNeuron(a=a)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("b", [0.2, 0.5, 1.0])
    def test_b_adaptation_strength(self, b: float) -> None:
        """The model runs across adaptation strengths b and its spikes are counted."""
        n = NonlinearLIFNeuron(b=b)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("tau_w", [50.0, 100.0, 200.0])
    def test_tau_w_sweep(self, tau_w: float) -> None:
        """The model is exercised across a sweep of tau_w."""
        n = NonlinearLIFNeuron(tau_w=tau_w)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.w)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = NonlinearLIFNeuron(dt=dt)
        for _ in range(10_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)
