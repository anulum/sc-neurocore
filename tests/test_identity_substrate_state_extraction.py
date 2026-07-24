# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStateExtraction from former test_identity_substrate.py

"""Focused suite: TestStateExtraction from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403


class TestStateExtraction:
    def test_extract_state_empty(self):
        sub = _make_substrate()
        state = sub.extract_state()
        assert "firing_rates" in state
        assert "dominant_patterns" in state
        assert state["total_steps"] == 0

    def test_extract_state_after_run(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        state = sub.extract_state()
        assert state["firing_rates"].shape[0] > 0
        assert state["total_steps"] == 100
