# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATAdaptation from former test_model_mat.py

"""Focused suite: TestMATAdaptation from former test_model_mat.py."""

from __future__ import annotations

from tests.model_mat_support import *


class TestMATAdaptation:
    def test_adaptation_reduces_rate(self):
        """Adaptation → first half has more spikes than second half."""
        n = SCResettingMATNeuron()
        s1 = sum(n.step(40.0) for _ in range(2500))
        s2 = sum(n.step(40.0) for _ in range(2500))
        assert s1 >= s2

    def test_adaptation_recovers(self):
        """After silence, thetas decay → threshold drops → fires again."""
        n = SCResettingMATNeuron()
        # Drive to adapt
        for _ in range(5000):
            n.step(40.0)
        theta_adapted = n.theta1 + n.theta2
        # Rest (let adaptation decay)
        for _ in range(2000):
            n.step(0.0)
        theta_recovered = n.theta1 + n.theta2
        assert theta_recovered < theta_adapted

    def test_theta_accumulation_with_bursts(self):
        """Rapid spiking accumulates theta beyond single h1/h2."""
        n = SCResettingMATNeuron()
        for _ in range(1000):
            n.step(50.0)
        # After sustained drive, thetas accumulate from multiple spikes
        assert n.theta2 > n.h2  # slow theta accumulates
