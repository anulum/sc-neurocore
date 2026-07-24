# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseConnectivity from former test_identity_director.py

"""Focused suite: TestDiagnoseConnectivity from former test_identity_director.py."""

from __future__ import annotations

from tests.identity_director_support import *  # noqa: F403


class TestDiagnoseConnectivity:
    def test_connectivity_too_dense(self):
        sub = _make_substrate()
        # Force dense connectivity
        sub.proj_ee.data[:] = 1.0
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "connectivity_too_dense" in problems

    def test_connectivity_too_sparse(self):
        sub = _make_substrate()
        # Force sparse connectivity
        sub.proj_ee.data[:] = 0.0
        sub.proj_ee.data[0] = 0.01  # keep one nonzero so density < 0.05
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "connectivity_too_sparse" in problems
