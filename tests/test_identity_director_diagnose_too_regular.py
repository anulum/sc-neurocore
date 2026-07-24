# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseTooRegular from former test_identity_director.py

"""Focused suite: TestDiagnoseTooRegular from former test_identity_director.py."""

from __future__ import annotations

from tests.identity_director_support import *  # noqa: F403


class TestDiagnoseTooRegular:
    def test_too_regular(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 0.1,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "too_regular" in problems
