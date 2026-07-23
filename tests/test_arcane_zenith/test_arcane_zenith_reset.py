# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReset from former test_arcane_zenith.py

"""Focused suite: TestReset from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403

class TestReset:
    def test_reset_clears_fast_and_working_compartments(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(50):
            core.step(5.0)
        core.reset()
        assert core.neuron.v_fast == 0.0
        assert core.neuron.v_work == 0.0

    def test_reset_preserves_identity_deep_compartment(self):
        # v_deep is the identity of the neuron; reset() must not clear it.
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(200):
            core.step(2.5)
        v_deep_before = core.neuron.v_deep
        core.reset()
        assert core.neuron.v_deep == v_deep_before

    def test_reset_zeroes_identity_drift_accumulator(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(50):
            core.step(5.0)
        core.reset()
        assert core.neuron.identity_drift == 0.0
