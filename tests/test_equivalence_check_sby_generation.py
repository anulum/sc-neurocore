# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSbyGeneration from former test_equivalence_check.py

"""Focused suite: TestSbyGeneration from former test_equivalence_check.py."""

from __future__ import annotations

from tests.equivalence_check_support import *  # noqa: F403

class TestSbyGeneration:
    """The generated .sby script (no run required)."""

    def test_sby_reads_all_sources_and_sets_mode_depth(self) -> None:
        sby = equivalence_check._generate_sby(
            "equiv_miter",
            ["equiv_miter.v", "tiny_ref.v", "tiny_dut.v"],
            depth=12,
            mode="bmc",
            engine="z3",
        )
        assert "mode bmc" in sby
        assert "bmc: depth 12" in sby
        assert "smtbmc z3" in sby
        assert "read -formal equiv_miter.v" in sby
        assert "prep -top equiv_miter" in sby

    def test_verdict_parsing_picks_last_done_line(self) -> None:
        stdout = "DONE (ERROR, rc=16)\nDONE (PASS, rc=0)\n"
        assert equivalence_check._parse_verdict(stdout) == ("PASS", 0)

    def test_verdict_parsing_unknown_when_absent(self) -> None:
        assert equivalence_check._parse_verdict("no summary here") == ("UNKNOWN", -1)
