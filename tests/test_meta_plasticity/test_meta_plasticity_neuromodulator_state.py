# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorState from former test_meta_plasticity.py

"""Focused suite: TestNeuromodulatorState from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestNeuromodulatorState:
    def test_initial_levels(self):
        ns = NeuromodulatorState()
        for nm in NeuromodulatorType:
            assert ns.levels[nm] == 0.5

    def test_update_high_surprise(self):
        ns = NeuromodulatorState()
        ns.update(novelty=0.5, surprise=1.0, gci=0.5)
        assert ns.levels[NeuromodulatorType.DOPAMINE] > 0.5

    def test_update_bounded(self):
        ns = NeuromodulatorState()
        for _ in range(100):
            ns.update(novelty=1.0, surprise=1.0, gci=1.0)
        for nm in NeuromodulatorType:
            assert 0.0 <= ns.levels[nm] <= 1.0

    def test_modulation_factor_lr(self):
        ns = NeuromodulatorState()
        ns.levels[NeuromodulatorType.DOPAMINE] = 0.9
        assert ns.modulation_factor("lr") > 1.0

    def test_modulation_factor_default(self):
        ns = NeuromodulatorState()
        assert ns.modulation_factor("unknown") == 1.0

    def test_modulation_factor_tau(self):
        ns = NeuromodulatorState()
        # 0.8 + 0.4*(1 - ach=0.5) = 1.0
        assert ns.modulation_factor("tau") == pytest.approx(1.0)

    def test_modulation_factor_gain(self):
        ns = NeuromodulatorState()
        # 0.5 + 0.5*ne=0.5 = 0.75
        assert ns.modulation_factor("gain") == pytest.approx(0.75)
