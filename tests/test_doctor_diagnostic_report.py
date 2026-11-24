# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnosticReport from former test_doctor.py

"""Focused suite: TestDiagnosticReport from former test_doctor.py."""

from __future__ import annotations

from sc_neurocore.doctor import (
    Diagnosis,
    DiagnosticReport,
)
from sc_neurocore.doctor.diagnose import (
    Severity,
)


class TestDiagnosticReport:
    """The diagnostic report."""

    def test_empty(self) -> None:
        """An empty report scores 100 with no critical finding."""
        r = DiagnosticReport(target="ice40")
        assert r.score == 100
        assert r.has_critical is False

    def test_score_with_findings(self) -> None:
        """A critical finding lowers the score."""
        r = DiagnosticReport(
            target="ice40",
            findings=[
                Diagnosis("test", Severity.WARNING, "msg", "fix"),
                Diagnosis("test", Severity.CRITICAL, "msg", "fix"),
            ],
        )
        assert r.score < 100
        assert r.has_critical is True

    def test_summary(self) -> None:
        """The summary names the device, findings and advice."""
        r = DiagnosticReport(
            target="artix7",
            findings=[Diagnosis("hw", Severity.WARNING, "high util", "prune")],
        )
        s = r.summary()
        assert "artix7" in s
        assert "high util" in s
        assert "prune" in s
