# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExperienceInjection from former test_identity_substrate.py

"""Focused suite: TestExperienceInjection from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403


class TestExperienceInjection:
    def test_inject_changes_weights(self):
        sub = _make_substrate()
        weights_before = sub.ee_weights.copy()
        sub.inject_experience("The system decided to increase inhibition based on rate analysis.")
        weights_after = sub.ee_weights
        assert not np.array_equal(weights_before, weights_after)

    def test_inject_increases_step_count(self):
        sub = _make_substrate()
        steps_before = sub._total_steps
        sub.inject_experience("Short trace.")
        assert sub._total_steps > steps_before
