# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHardwareBenchmark from former test_hardware.py

"""Focused suite: TestHardwareBenchmark from former test_hardware.py."""

from __future__ import annotations

from tests.hardware_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestHardwareBenchmark:
    def test_map_10k_neurons(self):
        """Map 10,000 neurons to Loihi in < 2 seconds."""
        rng = np.random.default_rng(42)
        adj = (rng.random((10_000, 10_000)) > 0.999).astype(float)
        np.fill_diagonal(adj, 0)
        mapper = Mapper()
        device = get_device(DeviceFamily.LOIHI)
        t0 = time.perf_counter()
        placements = mapper.map_greedy(adj, device)
        elapsed = time.perf_counter() - t0
        assert len(placements) == 10_000
        assert_load_tolerant_throughput(
            label="hardware mapping run",
            observed_per_second=1.0 / elapsed,
            strict_minimum_per_second=0.5,
        )

    def test_estimate_10k_neurons(self):
        """Resource estimation for 10k neurons in < 1 second."""
        rng = np.random.default_rng(42)
        adj = (rng.random((10_000, 10_000)) > 0.999).astype(float)
        np.fill_diagonal(adj, 0)
        estimator = ResourceEstimator()
        t0 = time.perf_counter()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        elapsed = time.perf_counter() - t0
        assert result.neurons_mapped == 10_000
        assert_load_tolerant_throughput(
            label="hardware estimation run",
            observed_per_second=1.0 / elapsed,
            strict_minimum_per_second=1.0,
        )

    def test_constraint_check_1k_neurons(self):
        """Constraint check for 1000 neurons in < 0.5 seconds."""
        rng = np.random.default_rng(42)
        adj = (rng.random((1000, 1000)) > 0.99).astype(float)
        np.fill_diagonal(adj, 0)
        checker = ConstraintChecker()
        constraints = HardwareConstraints.from_device(get_device(DeviceFamily.LOIHI))
        t0 = time.perf_counter()
        violations = checker.check(adj, constraints)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="hardware constraint run",
            observed_per_second=1.0 / elapsed,
            strict_minimum_per_second=2.0,
        )
