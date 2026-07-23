# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaThreshold from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaThreshold from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403

class TestSigmaDeltaThreshold:
    def test_lower_threshold_higher_rate(self):
        n_low = SigmaDeltaNeuron(v_threshold=0.5)
        n_high = SigmaDeltaNeuron(v_threshold=2.0)
        s_low = sum(1 for _ in range(1000) if n_low.step(0.3) == 1)
        s_high = sum(1 for _ in range(1000) if n_high.step(0.3) == 1)
        assert s_low > s_high

    def test_threshold_controls_quantisation_step(self):
        """Each spike represents ±θ of accumulated signal."""
        n = SigmaDeltaNeuron(v_threshold=2.0)
        # I=0.5 → rate = 0.5/2.0 = 0.25 spikes/step
        outputs = [n.step(0.5) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * 0.5 / 2.0
        assert abs(pos - expected) <= 2

    def test_very_small_threshold(self):
        """θ → 0: almost every step produces a spike."""
        n = SigmaDeltaNeuron(v_threshold=0.01)
        outputs = [n.step(0.1) for _ in range(100)]
        pos = outputs.count(1)
        assert pos >= 90  # rate = 0.1/0.01 = 10, but max 1/step → 100
