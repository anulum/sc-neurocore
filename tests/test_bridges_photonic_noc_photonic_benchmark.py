# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhotonicBenchmark from former test_bridges_photonic_noc.py

"""Focused suite: TestPhotonicBenchmark from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestPhotonicBenchmark:
    def test_compile_50_node_network(self):
        """End-to-end 50-node SC → photonic compile."""
        compiler = SCToPhotonic()
        rng = np.random.default_rng(42)
        adj = (rng.random((50, 50)) > 0.85).astype(float)
        np.fill_diagonal(adj, 0)
        t0 = time.perf_counter()
        design = compiler.compile(adj)
        elapsed = time.perf_counter() - t0
        assert design.n_nodes == 50
        assert_load_tolerant_throughput(
            label="photonic network compilation",
            observed_per_second=50 / elapsed,
            strict_minimum_per_second=5.0,
        )

    def test_wdm_assignment_throughput(self):
        """WDM assignment for 100 signals."""
        assigner = WDMAssigner(max_channels=128)
        signals = [f"sig_{i}" for i in range(100)]
        t0 = time.perf_counter()
        channels = assigner.assign(signals)
        elapsed = time.perf_counter() - t0
        assert len(channels) == 100
        assert_load_tolerant_throughput(
            label="WDM assignment",
            observed_per_second=100 / elapsed,
            strict_minimum_per_second=100.0,
        )
