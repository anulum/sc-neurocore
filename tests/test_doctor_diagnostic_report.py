# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnosticReport from former test_doctor.py

"""Focused suite: TestDiagnosticReport from former test_doctor.py."""

from __future__ import annotations

from tests.doctor_support import *  # noqa: F403


class TestDiagnosticReport:
    def test_empty(self):
        r = DiagnosticReport(target="ice40")
        assert r.score == 100
        assert r.has_critical is False

    def test_score_with_findings(self):
        r = DiagnosticReport(
            target="ice40",
            findings=[
                Diagnosis("test", Severity.WARNING, "msg", "fix"),
                Diagnosis("test", Severity.CRITICAL, "msg", "fix"),
            ],
        )
        assert r.score < 100
        assert r.has_critical is True

    def test_summary(self):
        r = DiagnosticReport(
            target="artix7",
            findings=[Diagnosis("hw", Severity.WARNING, "high util", "prune")],
        )
        s = r.summary()
        assert "artix7" in s
        assert "high util" in s
        assert "prune" in s
