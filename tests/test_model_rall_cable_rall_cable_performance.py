# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCablePerformance from former test_model_rall_cable.py

"""Focused suite: TestRallCablePerformance from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403


class TestRallCablePerformance:
    def test_isolation_throughput(self) -> None:
        import time

        n = RallCableNeuron(n_comp=5)
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # N-compartment cable with numpy array ops
        assert rate > 10_000, f"isolation: {rate:.0f} steps/s"
