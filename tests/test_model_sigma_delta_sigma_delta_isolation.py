# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaIsolation from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaIsolation from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403

class TestSigmaDeltaIsolation:
    def test_construction_defaults(self):
        n = SigmaDeltaNeuron()
        assert n.sigma == 0.0
        assert n.v_threshold == 1.0

    def test_step_returns_ternary(self):
        """Output is {-1, 0, +1}, not just binary."""
        n = SigmaDeltaNeuron()
        assert n.step(0.0) == 0
        n2 = SigmaDeltaNeuron()
        assert n2.step(1.5) == 1  # sigma=1.5 ≥ 1.0 → +1
        n3 = SigmaDeltaNeuron()
        assert n3.step(-1.5) == -1  # sigma=-1.5 ≤ -1.0 → -1

    def test_sigma_accumulates(self):
        """Input is summed into sigma (integration)."""
        n = SigmaDeltaNeuron(v_threshold=100.0)  # high threshold to avoid spikes
        for _ in range(10):
            n.step(0.3)
        assert abs(n.sigma - 3.0) < 1e-10

    def test_reset(self):
        n = SigmaDeltaNeuron()
        for _ in range(50):
            n.step(0.5)
        n.reset()
        assert n.sigma == 0.0
