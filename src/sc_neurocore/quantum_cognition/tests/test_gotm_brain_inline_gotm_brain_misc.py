# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGOTMBrainMisc from former test_gotm_brain_inline.py

"""Focused suite: TestGOTMBrainMisc from former test_gotm_brain_inline.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

from gotm_brain_inline_support import *  # noqa: F403


class TestGOTMBrainMisc:
    def test_reset(self) -> None:
        brain = GOTMBrain(n_neurons=4)
        chunk = ContentChunk(
            repo_name="t",
            file_path="t.md",
            chunk_index=0,
            text="content",
            content_type="markdown",
            weight=1.0,
        )
        brain.learn_step(chunk, np.ones(4))
        assert brain._total_steps > 0
        brain.reset()
        assert brain._total_steps == 0
        assert len(brain._history) == 0

    def test_get_learning_state(self) -> None:
        brain = GOTMBrain(n_neurons=4)
        state = brain.get_learning_state()
        assert state["n_neurons"] == 4
        assert "avg_atp" in state
        assert "avg_entanglement" in state

    def test_repr(self) -> None:
        brain = GOTMBrain(n_neurons=8)
        r = repr(brain)
        assert "GOTMBrain" in r
        assert "8" in r

    def test_learning_step_to_dict(self) -> None:
        step = LearningStep(
            step_index=0,
            directive="FOCUS",
            target_coherence=0.8,
            n_spikes=3,
            avg_atp=0.95,
            avg_entanglement=0.125,
            chunk_summary="test",
            chunk_sha256="abc123",
        )
        d = step.to_dict()
        assert d["directive"] == "FOCUS"
        assert d["n_spikes"] == 3
