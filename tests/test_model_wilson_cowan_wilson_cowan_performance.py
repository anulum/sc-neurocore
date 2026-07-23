# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanPerformance from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanPerformance from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403

class TestWilsonCowanPerformance:
    def test_isolation_runtime_regression_sentinel(self):
        """Bound pathological slowdowns without making CI throughput claims."""
        n = WilsonCowanUnit()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0
        assert np.isfinite(n.e) and np.isfinite(n.i)
