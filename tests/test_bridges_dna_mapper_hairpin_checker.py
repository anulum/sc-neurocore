# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHairpinChecker from former test_bridges_dna_mapper.py

"""Focused suite: TestHairpinChecker from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403


class TestHairpinChecker:
    def test_check_strand(self) -> None:
        checker = HairpinChecker()
        result = checker.check_strand("GCGATCGC")
        assert isinstance(result, list)
