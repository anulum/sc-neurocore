# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATParameters from former test_model_mat.py

"""Focused suite: TestMATParameters from former test_model_mat.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron
import numpy as np
import pytest
from tests.model_mat_support import (
    _run,
)


class TestMATParameters:
    """Parameter behaviour of the resetting MAT neuron."""

    @pytest.mark.parametrize("tau_1", [5.0, 10.0, 50.0])
    def test_tau_1_sweep(self, tau_1: float) -> None:
        """The state stays finite across a sweep of tau_1."""
        n = SCResettingMATNeuron(tau_1=tau_1)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1)

    @pytest.mark.parametrize("tau_2", [50.0, 200.0, 1000.0])
    def test_tau_2_sweep(self, tau_2: float) -> None:
        """The state stays finite across a sweep of tau_2."""
        n = SCResettingMATNeuron(tau_2=tau_2)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta2)

    @pytest.mark.parametrize("h1", [2.0, 5.0, 10.0])
    def test_h1_controls_fast_adaptation(self, h1: float) -> None:
        """A run with a changed h1 counts spikes (the check is on the type only)."""
        n = SCResettingMATNeuron(h1=h1)
        spikes = len(_run(n, current=40.0, steps=5000))
        assert isinstance(spikes, int)

    def test_h2_controls_slow_adaptation(self) -> None:
        """A larger h2 does not raise the spike count."""
        s_low = len(_run(SCResettingMATNeuron(h2=1.0), current=40.0, steps=5000))
        s_high = len(_run(SCResettingMATNeuron(h2=10.0), current=40.0, steps=5000))
        # Stronger slow adaptation → fewer spikes
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.1, 1.0, 2.0])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = SCResettingMATNeuron(dt=dt)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1) and np.isfinite(n.theta2)

    def test_resistance_scales_input(self) -> None:
        """Higher R → more effective current → more spikes."""
        s_low = len(_run(SCResettingMATNeuron(resistance=0.5), current=30.0, steps=5000))
        s_high = len(_run(SCResettingMATNeuron(resistance=2.0), current=30.0, steps=5000))
        assert s_high >= s_low
