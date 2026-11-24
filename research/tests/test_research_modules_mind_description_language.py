# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMindDescriptionLanguage from former test_research_modules.py

"""Focused suite: TestMindDescriptionLanguage from former test_research_modules.py."""

from __future__ import annotations

import pytest

from sc_neurocore.core.mdl_parser import MindDescriptionLanguage


class TestMindDescriptionLanguage:
    """Encoding and decoding mind descriptions."""

    def test_encode_decode_roundtrip(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An encoded orchestrator decodes back to its agent name and module states."""

        class MockModule:
            """A module stand-in with a fixed weight state."""

            def get_state(self) -> dict[str, list[float]]:
                """Return the fixed weight state."""
                return {"w": [0.1, 0.2]}

        class MockOrchestrator:
            """An orchestrator stand-in holding one module."""

            modules = {"layer1": MockModule()}

        mdl_str = MindDescriptionLanguage.encode(MockOrchestrator(), "TestBot")
        assert "TestBot" in mdl_str
        assert "layer1" in mdl_str

        data = MindDescriptionLanguage.decode(mdl_str)
        assert data["agent_name"] == "TestBot"
        assert "layer1" in data["state"]

    def test_decode_minimal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A minimal description decodes to its agent name."""
        yaml_str = "agent_name: Min\nversion: '1.0'\narchitecture: {}\nstate: {}\n"
        data = MindDescriptionLanguage.decode(yaml_str)
        assert data["agent_name"] == "Min"
