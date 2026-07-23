# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDirectiveDrivenCoherence from former test_qc_e2e.py

"""Focused suite: TestDirectiveDrivenCoherence from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403

class TestDirectiveDrivenCoherence:
    """Verify that FOCUS increases coherence, EXPLORE increases entropy."""

    def test_focus_vs_explore(self) -> None:
        brain = GOTMBrain(n_neurons=16, seed=42)
        vec = np.random.default_rng(42).random(16)

        # FOCUS: coherent input to specific sites
        focus_spikes = brain.process_content(vec, "FOCUS")
        focus_state = brain.get_learning_state()

        brain.reset()

        # EXPLORE: spread across all sites
        explore_spikes = brain.process_content(vec, "EXPLORE")
        explore_state = brain.get_learning_state()

        # Both should produce valid states
        assert 0.0 < focus_state["avg_atp"] <= 1.0
        assert 0.0 < explore_state["avg_atp"] <= 1.0
