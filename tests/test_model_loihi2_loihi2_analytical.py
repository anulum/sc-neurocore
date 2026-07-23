# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2Analytical from former test_model_loihi2.py

"""Focused suite: TestLoihi2Analytical from former test_model_loihi2.py."""

from __future__ import annotations

from tests.model_loihi2_support import *  # noqa: F403

class TestLoihi2Analytical:
    def test_three_state_variables(self):
        n = Loihi2Neuron()
        for attr in ["s1", "s2", "s3"]:
            assert hasattr(n, attr)

    def test_s2_integrates_input(self):
        n = Loihi2Neuron()
        n.step(500)
        assert n.s2 > 0

    def test_s1_driven_by_s2(self):
        """s1 = s1 - s1//tau1 + w12·s2. w12=1 → s2 drives s1."""
        n = Loihi2Neuron()
        n.step(500)
        n.step(0)  # s2 still has residual → drives s1
        assert n.s1 > 0

    def test_spike_increments_s3(self):
        """On spike: s3 += s3_incr (adaptation)."""
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s3 >= n.s3_incr
                break

    def test_integer_division_decay(self):
        """Decay via integer division: s -= s // tau."""
        n = Loihi2Neuron()
        n.s1 = 100
        decay = 100 // n.tau1
        assert decay == 10  # 100 // 10

    def test_spike_resets_s1(self):
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s1 == n.s1_reset
                break
