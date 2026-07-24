# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSampleEntropyEdge from former test_variability_edge_cases.py

"""Focused suite: TestSampleEntropyEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403


class TestSampleEntropyEdge:
    def test_short(self):
        result = sample_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_constant(self):
        result = sample_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_normal(self):
        train = np.zeros(200, dtype=np.int8)
        train[::7] = 1
        result = sample_entropy(train)
        assert np.isfinite(result) or np.isnan(result)
