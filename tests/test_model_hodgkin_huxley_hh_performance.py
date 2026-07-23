# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHPerformance from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHPerformance from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

from tests.model_hodgkin_huxley_support import *  # noqa: F403

class TestHHPerformance:
    def test_isolation_throughput(self):
        """HH is slow due to 100 sub-steps + exp() per step."""
        n = HodgkinHuxleyNeuron()
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        steps_per_s = N / elapsed
        # Expected ~670 steps/s; just verify it's > 100
        assert steps_per_s > 100, f"{steps_per_s:.0f} steps/s"
