# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthLeakEffect from former test_model_truenorth.py

"""Focused suite: TestTrueNorthLeakEffect from former test_model_truenorth.py."""

from __future__ import annotations

from tests.model_truenorth_support import *  # noqa: F403

class TestTrueNorthLeakEffect:
    def test_leak_reduces_effective_rate(self):
        """Higher leak → lower effective current → fewer spikes."""
        n_noleak = TrueNorthNeuron(leak=0)
        n_leak = TrueNorthNeuron(leak=10)
        s_noleak = sum(n_noleak.step(20) for _ in range(1000))
        s_leak = sum(n_leak.step(20) for _ in range(1000))
        assert s_noleak > s_leak

    def test_analytical_rate_with_leak(self):
        """Rate = steps / ceil(θ / (I - leak)) when I > leak."""
        n = TrueNorthNeuron(leak=10, threshold=100)
        I = 20
        steps = 1000
        spikes = sum(n.step(I) for _ in range(steps))
        effective = I - n.leak  # 10
        expected = steps * effective // n.threshold
        assert abs(spikes - expected) <= 2
