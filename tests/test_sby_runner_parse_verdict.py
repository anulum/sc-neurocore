# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParseVerdict from former test_sby_runner.py

"""Focused suite: TestParseVerdict from former test_sby_runner.py."""

from __future__ import annotations

from tests.sby_runner_support import *  # noqa: F403


class TestParseVerdict:
    """Verdict extraction from ``sby`` output."""

    def test_picks_last_done_line(self) -> None:
        assert parse_verdict("DONE (ERROR, rc=16)\nDONE (PASS, rc=0)\n") == ("PASS", 0)

    def test_unknown_when_absent(self) -> None:
        assert parse_verdict("no summary here") == ("UNKNOWN", -1)

    def test_fail_verdict_with_code(self) -> None:
        assert parse_verdict("DONE (FAIL, rc=2)") == ("FAIL", 2)
