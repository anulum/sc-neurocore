# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPropertyProofResult from former test_formal_property_check.py

"""Focused suite: TestPropertyProofResult from former test_formal_property_check.py."""

from __future__ import annotations

from sc_neurocore.compiler.formal_property_check import (
    PropertyProofResult,
)


class TestPropertyProofResult:
    """The result dataclass."""

    def test_defaults(self) -> None:
        """A proof result has no counterexample, trace or summary by default."""
        result = PropertyProofResult(
            proven=True, verdict="PASS", mode="bmc", depth=8, engine="z3", returncode=0
        )
        assert result.counterexample is None
        assert result.trace_path is None
        assert result.summary == []
