# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLearningStep from former test_gotm_brain.py

"""Focused suite: TestLearningStep from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403


class TestLearningStep:
    def test_to_dict(self) -> None:
        s = LearningStep(
            step_index=0,
            directive="FOCUS",
            target_coherence=0.8,
            n_spikes=5,
            avg_atp=0.95,
            avg_entanglement=0.125,
            chunk_summary="test",
            chunk_sha256="abc123",
        )
        d = s.to_dict()
        assert d["step"] == 0
        assert d["directive"] == "FOCUS"
