# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLvrEdge from former test_variability_edge_cases.py

"""Focused suite: TestLvrEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403

class TestLvrEdge:
    def test_empty(self):
        result = lvr(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_regular(self):
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = lvr(train)
        assert np.isfinite(result)
