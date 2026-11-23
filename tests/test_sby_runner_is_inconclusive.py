# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsInconclusive from former test_sby_runner.py

"""Focused suite: TestIsInconclusive from former test_sby_runner.py."""

from __future__ import annotations

from sc_neurocore.compiler._sby_runner import (
    is_inconclusive,
    SbyRun,
)


class TestIsInconclusive:
    """The inconclusive k-induction signature (UNKNOWN, rc == 4)."""

    def test_unknown_rc4_is_inconclusive(self) -> None:
        """UNKNOWN with rc 4 is inconclusive."""
        assert is_inconclusive(SbyRun(verdict="UNKNOWN", rc=4, returncode=4)) is True

    def test_pass_is_not_inconclusive(self) -> None:
        """A PASS verdict is conclusive."""
        assert is_inconclusive(SbyRun(verdict="PASS", rc=0, returncode=0)) is False

    def test_error_is_not_inconclusive(self) -> None:
        """An ERROR verdict is conclusive."""
        assert is_inconclusive(SbyRun(verdict="ERROR", rc=16, returncode=16)) is False

    def test_unknown_without_rc4_is_not_inconclusive(self) -> None:
        """UNKNOWN without rc 4 is not inconclusive."""
        # A crashed run with no DONE line parses to (UNKNOWN, -1) — a tool failure,
        # not an inconclusive proof.
        assert is_inconclusive(SbyRun(verdict="UNKNOWN", rc=-1, returncode=1)) is False
