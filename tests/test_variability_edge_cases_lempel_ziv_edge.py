# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLempelZivEdge from former test_variability_edge_cases.py

"""Focused suite: TestLempelZivEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403

class TestLempelZivEdge:
    def test_empty(self):
        result = lempel_ziv_complexity(np.zeros(50, dtype=np.int8))
        assert np.isfinite(result)

    def test_all_ones(self):
        result = lempel_ziv_complexity(np.ones(50, dtype=np.int8))
        assert np.isfinite(result)

    def test_alternating(self):
        train = np.zeros(100, dtype=np.int8)
        train[::2] = 1
        result = lempel_ziv_complexity(train)
        assert np.isfinite(result)
