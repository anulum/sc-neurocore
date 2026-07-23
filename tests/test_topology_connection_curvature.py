# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnectionCurvature from former test_topology.py

"""Focused suite: TestConnectionCurvature from former test_topology.py."""

from __future__ import annotations

from tests.topology_support import *  # noqa: F403

class TestConnectionCurvature:
    def test_synchronized_full_coupling(self):
        phases = np.zeros(3)
        knm = np.ones((3, 3))
        F = connection_curvature(phases, knm)
        # cos(0) = 1 everywhere, so F = knm
        np.testing.assert_allclose(F, knm)

    def test_anti_phase_negative(self):
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        F = connection_curvature(phases, knm)
        # cos(pi) = -1, so F[0,1] = -1
        np.testing.assert_allclose(F[0, 1], -1.0)

    def test_diagonal_is_coupling_weighted(self):
        phases = np.array([0.0, 0.0, 0.0])
        knm = np.diag([1.0, 2.0, 3.0])
        F = connection_curvature(phases, knm)
        np.testing.assert_allclose(np.diag(F), [1.0, 2.0, 3.0])
