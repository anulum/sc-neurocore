# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSheafConsistencyDefect from former test_topology.py

"""Focused suite: TestSheafConsistencyDefect from former test_topology.py."""

from __future__ import annotations

from tests.topology_support import *  # noqa: F403

class TestSheafConsistencyDefect:
    def test_synchronized_is_zero(self):
        phases = np.zeros(5)
        knm = np.ones((5, 5))
        assert sheaf_consistency_defect(phases, knm) == 0.0

    def test_anti_phase_is_positive(self):
        phases = np.array([0.0, np.pi, 0.0, np.pi])
        knm = np.ones((4, 4))
        defect = sheaf_consistency_defect(phases, knm)
        assert defect > 0.0

    def test_zero_coupling_is_zero(self):
        phases = np.random.rand(5) * 2 * np.pi
        knm = np.zeros((5, 5))
        assert sheaf_consistency_defect(phases, knm) == 0.0
