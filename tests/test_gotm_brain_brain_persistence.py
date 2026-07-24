# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrainPersistence from former test_gotm_brain.py

"""Focused suite: TestBrainPersistence from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403


class TestBrainPersistence:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """save_state → load_state preserves full brain state."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated", seed=42)
        chunk = ContentChunk("R", "a.py", 0, "persistence test content", "code", 1.0)
        vec = np.random.default_rng(42).random(8)
        brain.learn_step(chunk, vec)
        state_before = brain.get_learning_state()

        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        brain2 = GOTMBrain(n_neurons=8, bridge_backend="emulated", seed=99)
        brain2.load_state(path)
        state_after = brain2.get_learning_state()

        assert state_before["total_steps"] == state_after["total_steps"]
        assert state_before["total_spikes"] == state_after["total_spikes"]
        assert state_after["history_length"] == 1

    def test_load_dimension_mismatch(self, tmp_path: Path) -> None:
        """load_state raises ValueError on neuron count mismatch."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        brain2 = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        with pytest.raises(ValueError, match="neurons"):
            brain2.load_state(path)

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """Saved file is valid JSON."""
        import json

        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        with open(path) as f:
            data = json.load(f)
        assert data["n_neurons"] == 4
        assert "neuron_states" in data
        assert "pool_state" in data
        assert "history" in data
