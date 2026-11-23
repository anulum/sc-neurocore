# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMuMax3Parser from former test_spintronic_mapper.py

"""Focused suite: TestMuMax3Parser from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    MuMax3OutputParser,
    MuMax3Result,
)


class TestMuMax3Parser:
    """Parsing MuMax3 output tables."""

    def test_parse_table(self) -> None:
        """A table ending in negative mz is a completed switch."""
        table = "# t mx my mz\n5e-9\t0.01\t0.02\t-0.99"
        result = MuMax3OutputParser.parse_table(table)
        assert result.switched is True
        assert result.final_mz < 0

    def test_successful_switch(self) -> None:
        """A switched result counts as a successful switch."""
        r = MuMax3Result(0.01, 0.02, -0.99, True)
        assert MuMax3OutputParser.is_switching_successful(r)

    def test_failed_switch(self) -> None:
        """An unswitched result is not a successful switch."""
        r = MuMax3Result(0.01, 0.02, 0.99, False)
        assert not MuMax3OutputParser.is_switching_successful(r)

    def test_empty_input(self) -> None:
        """Empty output gives a final mz of 0."""
        result = MuMax3OutputParser.parse_table("")
        assert result.final_mz == 0.0

    def test_parse_table_whitespace_separated_row(self) -> None:
        """Whitespace-separated rows parse like tab-separated ones."""
        # A row that is space- rather than tab-separated falls back to a generic
        # whitespace split and still parses.
        table = "# t mx my mz\n5e-9 0.01 0.02 -0.99"
        result = MuMax3OutputParser.parse_table(table)
        assert result.switched is True
        assert result.final_mz < 0

    def test_parse_table_non_numeric_row_returns_default(self) -> None:
        """A non-numeric row gives the default unswitched result."""
        # A malformed row that cannot be parsed as floats yields a default
        # result rather than raising.
        result = MuMax3OutputParser.parse_table("# header\nnot a number row")
        assert result.final_mz == 0.0
        assert result.switched is False
