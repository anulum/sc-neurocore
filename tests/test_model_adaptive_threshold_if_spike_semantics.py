# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSemantics from former test_model_adaptive_threshold_if.py

"""Focused suite: TestSpikeSemantics from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.adaptive_threshold_if import (
    AdaptiveThresholdIFNeuron,
)


class TestSpikeSemantics:
    """Candidate crossing, reset, fixed shift, and adaptation."""

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        n = AdaptiveThresholdIFNeuron()
        assert n.step(0.0) in (0, 1)

    def test_crossing_installs_reset_and_fixed_shift(self) -> None:
        """A crossing resets the voltage and shifts the threshold."""
        n = AdaptiveThresholdIFNeuron(v=-50.5, theta=-51.0)
        assert n.step(0.0) == 1
        assert n.v == -65.0
        relaxed = -50.0 + (-51.0 + 50.0) * np.exp(-0.1 / 50.0)
        assert n.theta == pytest.approx(relaxed + 5.0, rel=0.0, abs=1e-14)

    def test_shifted_threshold_does_not_immediately_retrigger(self) -> None:
        """The shifted threshold does not fire again at once."""
        n = AdaptiveThresholdIFNeuron(v=-50.5, theta=-51.0)
        assert n.step(0.0) == 1
        assert n.step(0.0) == 0

    def test_spikes_under_drive(self) -> None:
        """Strong drive produces spikes."""
        n = AdaptiveThresholdIFNeuron()
        spikes = sum(n.step(100.0) for _ in range(2000))
        assert spikes > 0, "no spikes at I=100"

    def test_threshold_adapts(self) -> None:
        """The threshold rises after spikes."""
        n = AdaptiveThresholdIFNeuron()
        theta_init = n.theta
        for _ in range(2000):
            n.step(100.0)
        assert n.theta > theta_init, "threshold did not increase after spiking"

    def test_adaptation_accumulates_per_spike_with_decay(self) -> None:
        """Adaptation adds up per spike and decays between them."""
        n = AdaptiveThresholdIFNeuron()
        first_spike = None
        for index in range(2000):
            if n.step(100.0) == 1:
                first_spike = index
                break
        assert first_spike is not None
        theta_after_first = n.theta
        for _ in range(2000):
            if n.step(100.0) == 1:
                break
        assert n.theta > theta_after_first - 5.0

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = AdaptiveThresholdIFNeuron()
        for _ in range(5000):
            n.step(200.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.theta)

    def test_reset_restores_documented_state_preserving_configuration(self) -> None:
        """A reset restores the documented state and keeps the configuration."""
        n = AdaptiveThresholdIFNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_rest
        assert (n.delta_theta, n.tau_m, n.tau_theta, n.dt) == (5.0, 10.0, 50.0, 0.1)
