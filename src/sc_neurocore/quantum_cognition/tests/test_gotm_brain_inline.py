# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Inline tests for GOTMBrain — persistence, LLM fallback, CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.content_indexer import ContentChunk
from sc_neurocore.quantum_cognition.gotm_brain import (
    HAS_LLM,
    GOTMBrain,
    LearningStep,
)


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
