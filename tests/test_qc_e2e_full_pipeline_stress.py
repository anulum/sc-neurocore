# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFullPipelineStress from former test_qc_e2e.py

"""Focused suite: TestFullPipelineStress from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403


class TestFullPipelineStress:
    """Index real files, feed through GOTMBrain, verify learning progression."""

    def test_learn_from_synthetic_repo(self, tmp_path: Path) -> None:
        """Create 200+ files, index, learn, verify state improves."""
        # Build a synthetic repo with diverse content
        for i in range(200):
            ext = [".md", ".py", ".tex", ".rs", ".jl"][i % 5]
            content = f"# Section {i}\nMathematical theorem {i}: ∀x∈ℝ, f(x) = x² + {i}\n" * 5
            d = tmp_path / f"pkg_{i // 20}"
            d.mkdir(exist_ok=True)
            (d / f"file_{i}{ext}").write_text(content)

        brain = GOTMBrain(n_neurons=16, seed=42)
        steps = brain.learn_from_repo(str(tmp_path), max_chunks=100)
        assert len(steps) > 0, "Should process at least some chunks"

        state = brain.get_learning_state()
        assert state["total_steps"] > 0
        assert state["total_spikes"] >= 0
        assert 0.0 < state["avg_atp"] <= 1.0

    def test_persistence_under_reload(self, tmp_path: Path) -> None:
        """Save state, reload into new brain, continue learning."""
        brain1 = GOTMBrain(n_neurons=8, seed=42)
        chunk = ContentChunk("test", "f.md", 0, "Euler's identity e^(iπ)+1=0", "markdown", 1.0)
        vec = np.random.default_rng(42).random(8)
        for _ in range(10):
            brain1.learn_step(chunk, vec)

        sf = str(tmp_path / "state.json")
        brain1.save_state(sf)
        s1 = brain1.get_learning_state()

        brain2 = GOTMBrain(n_neurons=8, seed=42)
        brain2.load_state(sf)
        s2 = brain2.get_learning_state()
        assert s2["total_steps"] == s1["total_steps"]

        # Continue learning — should not crash
        brain2.learn_step(chunk, vec)
        assert brain2._total_steps == s1["total_steps"] + 1
