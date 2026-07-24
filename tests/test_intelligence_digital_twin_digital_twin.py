# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDigitalTwin from former test_intelligence_digital_twin.py

"""Focused suite: TestDigitalTwin from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403


class TestDigitalTwin:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_digital_twin

        code = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
        assert "Twin" in code
        assert "def step" in code
        assert "def compare" in code
