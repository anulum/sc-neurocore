# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertPerformance from former test_model_siegert.py

"""Focused suite: TestSiegertPerformance from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403


class TestSiegertPerformance:
    def test_isolation_throughput(self) -> None:
        """Slow due to Gauss-Legendre quadrature (40 points)."""
        n = SiegertTransferFunction()
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 100  # ~524 steps/s
