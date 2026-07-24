# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRaiseForIncomplete from former test_sby_runner.py

"""Focused suite: TestRaiseForIncomplete from former test_sby_runner.py."""

from __future__ import annotations

from tests.sby_runner_support import *  # noqa: F403


class TestRaiseForIncomplete:
    """The verdict-completeness guard."""

    def test_pass_does_not_raise(self) -> None:
        raise_for_incomplete(SbyRun(verdict="PASS", rc=0, returncode=0), what="equivalence proof")

    def test_fail_does_not_raise(self) -> None:
        raise_for_incomplete(SbyRun(verdict="FAIL", rc=2, returncode=2), what="property proof")

    def test_inconclusive_does_not_raise(self) -> None:
        # A base-case-passed / induction-inconclusive k-induction is a real outcome.
        raise_for_incomplete(SbyRun(verdict="UNKNOWN", rc=4, returncode=4), what="property proof")

    def test_error_raises_with_label_and_tail(self) -> None:
        run = SbyRun(verdict="ERROR", rc=16, returncode=16, stdout="line1\nboom\n")
        with pytest.raises(RuntimeError, match="property proof did not complete"):
            raise_for_incomplete(run, what="property proof")

    def test_crash_without_done_line_raises(self) -> None:
        with pytest.raises(RuntimeError, match="verdict=UNKNOWN"):
            raise_for_incomplete(SbyRun(verdict="UNKNOWN", rc=-1, returncode=1), what="proof")
