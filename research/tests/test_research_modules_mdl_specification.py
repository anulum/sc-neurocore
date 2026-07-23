# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMDLSpecification from former test_research_modules.py

"""Focused suite: TestMDLSpecification from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestMDLSpecification:
    def test_defaults(self):
        spec = MDLSpecification()
        assert spec.version == "1.0"
        assert spec.agent_name == "Unknown"
        assert spec.architecture == {}

    def test_custom(self):
        spec = MDLSpecification(agent_name="TestAgent", version="2.0")
        assert spec.agent_name == "TestAgent"
