# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMindDescriptionLanguage from former test_research_modules.py

"""Focused suite: TestMindDescriptionLanguage from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestMindDescriptionLanguage:
    def test_encode_decode_roundtrip(self, capsys):
        class MockModule:
            def get_state(self):
                return {"w": [0.1, 0.2]}

        class MockOrchestrator:
            modules = {"layer1": MockModule()}

        mdl_str = MindDescriptionLanguage.encode(MockOrchestrator(), "TestBot")
        assert "TestBot" in mdl_str
        assert "layer1" in mdl_str

        data = MindDescriptionLanguage.decode(mdl_str)
        assert data["agent_name"] == "TestBot"
        assert "layer1" in data["state"]

    def test_decode_minimal(self, capsys):
        yaml_str = "agent_name: Min\nversion: '1.0'\narchitecture: {}\nstate: {}\n"
        data = MindDescriptionLanguage.decode(yaml_str)
        assert data["agent_name"] == "Min"
