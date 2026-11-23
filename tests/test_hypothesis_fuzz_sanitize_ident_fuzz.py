# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSanitizeIdentFuzz from former test_hypothesis_fuzz.py

"""Focused suite: TestSanitizeIdentFuzz from former test_hypothesis_fuzz.py."""

from __future__ import annotations

from hypothesis import (
    given,
    settings,
    strategies as st,
)
from sc_neurocore.hdl_gen._ident import (
    sanitize_ident,
)
from tests.hypothesis_fuzz_support import (
    _IDENT_STRINGS,
)


class TestSanitizeIdentFuzz:
    """Property: sanitize_ident always produces a valid Verilog identifier.

    It raises ValueError for empty or wholly invalid input.
    """

    @given(name=_IDENT_STRINGS)
    @settings(max_examples=500)
    def test_output_is_always_valid_verilog(self, name: str) -> None:
        """Whatever the input, the output is a valid Verilog identifier.

        It is alphanumeric plus underscore and does not start with a digit.
        """
        try:
            result = sanitize_ident(name)
        except ValueError:
            return  # Empty or completely invalid — acceptable

        # Verify result is a valid Verilog identifier
        assert len(result) > 0, "sanitize_ident returned empty string"
        assert result[0].isalpha() or result[0] == "_", (
            f"Identifier starts with invalid char: {result!r}"
        )
        for ch in result:
            assert ch.isalnum() or ch == "_", (
                f"Invalid character {ch!r} in sanitised identifier {result!r}"
            )

    @given(name=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,30}", fullmatch=True))
    @settings(max_examples=200)
    def test_valid_identifiers_pass_through(self, name: str) -> None:
        """Already-valid Verilog identifiers should pass through unchanged."""
        result = sanitize_ident(name)
        assert result == name
