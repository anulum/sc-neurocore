# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKLIFAnalytical from former test_model_klif.py

"""Focused suite: TestKLIFAnalytical from former test_model_klif.py."""

from __future__ import annotations

from tests.model_klif_support import *  # noqa: F403


class TestKLIFAnalytical:
    def test_v_update_formula(self):
        """V = α·V + k·I."""
        n = KLIFNeuron()
        v0 = n.v
        I = 0.5
        expected = n.alpha * v0 + n.k * I
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-12

    def test_k_scales_input(self):
        """k=2 → double effective input."""
        n1 = KLIFNeuron(k=1.0, v_threshold=100.0)
        n2 = KLIFNeuron(k=2.0, v_threshold=100.0)
        for _ in range(100):
            n1.step(1.0)
            n2.step(1.0)
        assert n2.v > n1.v

    def test_alpha_decay_without_input(self):
        """V decays by α per step when I=0."""
        n = KLIFNeuron(v_threshold=100.0)
        n.v = 0.5
        for _ in range(10):
            n.step(0.0)
        expected = 0.5 * n.alpha**10
        assert abs(n.v - expected) < 1e-10

    def test_spike_resets_voltage(self):
        n = KLIFNeuron()
        for _ in range(10_000):
            if n.step(1.0) == 1:
                assert n.v == n.v_reset
                break

    def test_zero_k_no_integration(self):
        """k=0 → V never integrates input."""
        n = KLIFNeuron(k=0.0)
        for _ in range(1000):
            n.step(10.0)
        # V = alpha^n * 0 = 0 (starts at 0, no input scaled)
        assert abs(n.v) < 1e-10
