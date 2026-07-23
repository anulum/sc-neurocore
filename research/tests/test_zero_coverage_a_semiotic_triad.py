# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSemioticTriad from former test_zero_coverage_a.py

"""Focused suite: TestSemioticTriad from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestSemioticTriad:
    def test_learn_and_interpret(self):
        from transcendent.noetic import SemioticTriad, Sign

        s = SemioticTriad()
        s.learn_association("fire", "heat")
        s.learn_association("heat", "pain")
        sign = Sign(signifier="flame", signified="fire", interpretant="danger")
        result = s.interpret(sign)
        assert result is not None

    def test_metaphor_distance(self):
        from transcendent.noetic import SemioticTriad

        s = SemioticTriad()
        s.learn_association("fire", "heat")
        s.learn_association("heat", "pain")
        d = s.metaphor_distance("fire", "pain", depth=5)
        assert isinstance(d, int)
