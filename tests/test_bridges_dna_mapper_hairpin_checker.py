# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHairpinChecker from former test_bridges_dna_mapper.py

"""Focused suite: TestHairpinChecker from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    HairpinChecker,
)


class TestHairpinChecker:
    """Smoke test of the hairpin check."""

    def test_check_strand(self) -> None:
        """The result is a list; its contents are not checked."""
        checker = HairpinChecker()
        result = checker.check_strand("GCGATCGC")
        assert isinstance(result, list)
