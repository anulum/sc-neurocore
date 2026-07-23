# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGOTMBrainLLM from former test_gotm_brain_inline.py

"""Focused suite: TestGOTMBrainLLM from former test_gotm_brain_inline.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from gotm_brain_inline_support import *  # noqa: F403

class TestGOTMBrainLLM:
    def test_fallback_directive(self) -> None:
        """Without LLM, should return STABILIZE."""
        brain = GOTMBrain(n_neurons=4)
        # If no LLM is available, fallback is STABILIZE
        if not HAS_LLM:
            d = brain.get_llm_guidance("test context")
            assert d == "STABILIZE"

    def test_process_content(self) -> None:
        """process_content should return list of spike indices."""
        brain = GOTMBrain(n_neurons=8, seed=42)
        vec = np.ones(8) * 0.5
        spikes = brain.process_content(vec, "FOCUS")
        assert isinstance(spikes, list)
        for s in spikes:
            assert 0 <= s < 8
