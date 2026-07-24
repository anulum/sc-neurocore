# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGOTMBrainPersistence from former test_gotm_brain_inline.py

"""Focused suite: TestGOTMBrainPersistence from former test_gotm_brain_inline.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

from gotm_brain_inline_support import *  # noqa: F403


class TestGOTMBrainPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """State should survive save → load cycle."""
        brain = GOTMBrain(n_neurons=8, seed=42)

        # Simulate some learning
        chunk = ContentChunk(
            repo_name="test",
            file_path="test.md",
            chunk_index=0,
            text="Test mathematical content",
            content_type="markdown",
            weight=1.0,
        )
        vec = np.random.default_rng(42).random(8)
        brain.learn_step(chunk, vec)

        state_file = str(tmp_path / "state.json")
        brain.save_state(state_file)
        assert Path(state_file).exists()

        # Load into a new brain
        brain2 = GOTMBrain(n_neurons=8, seed=42)
        brain2.load_state(state_file)
        assert brain2._total_steps == brain._total_steps
        assert len(brain2._history) == len(brain._history)

    def test_load_mismatched_neurons(self, tmp_path: Path) -> None:
        """Loading state with wrong neuron count should raise."""
        brain = GOTMBrain(n_neurons=8)
        state_file = str(tmp_path / "state.json")
        brain.save_state(state_file)

        brain2 = GOTMBrain(n_neurons=16)
        with pytest.raises(ValueError, match="neurons"):
            brain2.load_state(state_file)
