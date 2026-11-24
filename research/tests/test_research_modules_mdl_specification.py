# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMDLSpecification from former test_research_modules.py

"""Focused suite: TestMDLSpecification from former test_research_modules.py."""

from __future__ import annotations

from sc_neurocore.core.mdl_parser import MDLSpecification


class TestMDLSpecification:
    """The mind-description specification record."""

    def test_defaults(self) -> None:
        """The defaults are version 1.0, an unknown agent and an empty architecture."""
        spec = MDLSpecification()
        assert spec.version == "1.0"
        assert spec.agent_name == "Unknown"
        assert spec.architecture == {}

    def test_custom(self) -> None:
        """A custom agent name is kept."""
        spec = MDLSpecification(agent_name="TestAgent", version="2.0")
        assert spec.agent_name == "TestAgent"
