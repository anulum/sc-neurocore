# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaEdgeCases from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaEdgeCases from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaEdgeCases:
    def test_exact_threshold_crossing(self):
        """sigma exactly equals threshold → spike."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(1.0)
        assert s == 1
        assert abs(n.sigma) < 1e-10  # 1.0 - 1.0 = 0.0

    def test_exact_negative_threshold(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(-1.0)
        assert s == -1
        assert abs(n.sigma) < 1e-10

    def test_large_input_single_spike(self):
        """Even at I=100, only one spike per step (no multi-spike)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(100.0)
        assert s == 1  # only one +1, not 100

    def test_state_finite_long_run(self):
        """With I > θ, sigma grows unboundedly but stays finite."""
        n = SigmaDeltaNeuron()
        for _ in range(100000):
            n.step(2.0)
        assert np.isfinite(n.sigma)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SigmaDeltaNeuron()
            trace = [(n.step(0.3), n.sigma) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
