# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSemioticTriad from former test_zero_coverage_a.py

"""Focused suite: TestSemioticTriad from former test_zero_coverage_a.py."""

from __future__ import annotations

from typing import Any, cast


class TestSemioticTriad:
    """The semiotic triad research module."""

    def test_learn_and_interpret(self) -> None:
        """A learned sign is interpreted (the check is that the result is not None)."""
        from transcendent.noetic import SemioticTriad, Sign

        s = cast(Any, SemioticTriad)()
        s.learn_association("fire", "heat")
        s.learn_association("heat", "pain")
        sign = Sign(signifier="flame", signified="fire", interpretant="danger")
        result = s.interpret(sign)
        assert result is not None

    def test_metaphor_distance(self) -> None:
        """The metaphor distance is an integer (the check is on the type only)."""
        from transcendent.noetic import SemioticTriad

        s = cast(Any, SemioticTriad)()
        s.learn_association("fire", "heat")
        s.learn_association("heat", "pain")
        d = s.metaphor_distance("fire", "pain", depth=5)
        assert isinstance(d, int)
