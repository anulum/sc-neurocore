# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorThreshold from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorThreshold from former test_model_perfect_integrator.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.perfect_integrator import (
    PerfectIntegratorNeuron,
)


class TestPerfectIntegratorThreshold:
    """Threshold, reset, and spike timing."""

    def test_exact_threshold_fires(self) -> None:
        """The preserved SC profile fires when V reaches threshold exactly."""
        # Use I=10 so dV=1.0/step → hits threshold=1.0 exactly at step 1
        n = PerfectIntegratorNeuron(dt=0.1, c_m=1.0, v_threshold=1.0)
        s = n.step(10.0)
        assert s == 1

    def test_source_exact_threshold_waits_for_strict_crossing(self) -> None:
        """Naud-Gerstner's written > rule retains equality until the next drive."""
        n = PerfectIntegratorNeuron.naud_gerstner_2012()
        assert [n.step(5.0) for _ in range(3)] == [0, 0, 1]
        assert n.v == n.v_reset

    def test_reset_to_v_reset(self) -> None:
        """A spike resets the voltage to v_reset."""
        n = PerfectIntegratorNeuron()
        n.step(10.0)  # spike
        assert n.v == n.v_reset

    def test_custom_reset_potential(self) -> None:
        """Custom reset potential."""
        n = PerfectIntegratorNeuron(v_reset=-0.5, v_threshold=1.0)
        n.step(10.0)  # spike at v=1.0
        assert n.v == -0.5

    def test_superthreshold_instant_spike(self) -> None:
        """Very large current → spike on first step."""
        n = PerfectIntegratorNeuron()
        assert n.step(100.0) == 1
