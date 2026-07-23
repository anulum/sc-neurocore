# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptimalKernelBandwidthEdge from former test_variability_edge_cases.py

"""Focused suite: TestOptimalKernelBandwidthEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403

class TestOptimalKernelBandwidthEdge:
    def test_empty(self):
        result = optimal_kernel_bandwidth(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = optimal_kernel_bandwidth(train)
        assert np.isnan(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::10] = 1
        result = optimal_kernel_bandwidth(train)
        assert np.isfinite(result)
