# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseSilent from former test_identity_director.py

"""Focused suite: TestDiagnoseSilent from former test_identity_director.py."""

from __future__ import annotations

from tests.identity_director_support import *  # noqa: F403

class TestDiagnoseSilent:
    def test_silent(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 0.0,
                "cv": float("nan"),
                "fano": float("nan"),
                "perm_entropy": float("nan"),
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "silent" in problems
