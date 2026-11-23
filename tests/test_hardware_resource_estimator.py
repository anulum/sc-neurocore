# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResourceEstimator from former test_hardware.py

"""Focused suite: TestResourceEstimator from former test_hardware.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.hardware import (
    DeviceFamily,
    get_device,
    ResourceEstimate,
    ResourceEstimator,
)


class TestResourceEstimator:
    """Estimating device resources for a network."""

    def _make_adj(self, n: int, density: float = 0.1, seed: int = 42) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) < density).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def test_estimate_small_network(self) -> None:
        """A small network needs at least one core."""
        adj = self._make_adj(10, density=0.5)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert isinstance(result, ResourceEstimate)
        assert result.cores_needed >= 1
        assert result.neurons_mapped == 10
        assert result.fits is True

    def test_estimate_large_network(self) -> None:
        """A large network maps every neuron and draws power."""
        adj = self._make_adj(1000, density=0.01)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert result.neurons_mapped == 1000
        assert result.power_mw > 0

    def test_fits_small(self) -> None:
        """A small network fits the device."""
        adj = self._make_adj(10)
        estimator = ResourceEstimator()
        assert estimator.fits(adj, get_device(DeviceFamily.LOIHI)) is True

    def test_compare_devices(self) -> None:
        """Three devices are compared."""
        adj = self._make_adj(100, density=0.05)
        estimator = ResourceEstimator()
        devices = [
            get_device(f) for f in [DeviceFamily.LOIHI, DeviceFamily.SPINNAKER, DeviceFamily.AKIDA]
        ]
        results = estimator.compare(adj, devices)
        assert len(results) == 3
        assert all(isinstance(r, ResourceEstimate) for r in results)

    def test_utilization_range(self) -> None:
        """The utilisation lies in (0, 100]."""
        adj = self._make_adj(50)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert 0 < result.utilization_pct <= 100

    def test_latency_positive(self) -> None:
        """The latency is positive."""
        adj = self._make_adj(10)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert result.latency_us > 0
