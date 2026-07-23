# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReport from former test_identity_director.py

"""Focused suite: TestReport from former test_identity_director.py."""

from __future__ import annotations

from tests.identity_director_support import *  # noqa: F403

class TestReport:
    def test_report_healthy(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=[]):
            report = director.report()
        assert "healthy" in report

    def test_report_problems(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["rate_too_high", "bursty"]):
            report = director.report()
        assert "rate_too_high" in report
        assert "bursty" in report
