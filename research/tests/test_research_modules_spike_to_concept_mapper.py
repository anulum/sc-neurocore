# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeToConceptMapper from former test_research_modules.py

"""Focused suite: TestSpikeToConceptMapper from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestSpikeToConceptMapper:
    def test_active_concepts(self):
        mapper = SpikeToConceptMapper({0: "Vision", 1: "Motor", 2: "Audio"})
        spikes = np.array([1, 0, 1, 0])
        result = mapper.explain(spikes)
        assert "Vision" in result
        assert "Audio" in result

    def test_idle(self):
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 0, 0])
        assert "idle" in mapper.explain(spikes)

    def test_unknown_neuron(self):
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 1])  # neuron 1 not in map
        result = mapper.explain(spikes)
        assert "Unknown(1)" in result
