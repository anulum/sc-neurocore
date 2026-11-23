# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorState from former test_meta_plasticity.py

"""Focused suite: TestNeuromodulatorState from former test_meta_plasticity.py."""

from __future__ import annotations

import pytest
from sc_neurocore.meta_plasticity.meta_plasticity import (
    NeuromodulatorState,
    NeuromodulatorType,
)


class TestNeuromodulatorState:
    """Neuromodulator levels and their modulation."""

    def test_initial_levels(self) -> None:
        """Every neuromodulator starts at one half."""
        ns = NeuromodulatorState()
        for nm in NeuromodulatorType:
            assert ns.levels[nm] == 0.5

    def test_update_high_surprise(self) -> None:
        """High surprise raises dopamine."""
        ns = NeuromodulatorState()
        ns.update(novelty=0.5, surprise=1.0, gci=0.5)
        assert ns.levels[NeuromodulatorType.DOPAMINE] > 0.5

    def test_update_bounded(self) -> None:
        """Every level stays between zero and one."""
        ns = NeuromodulatorState()
        for _ in range(100):
            ns.update(novelty=1.0, surprise=1.0, gci=1.0)
        for nm in NeuromodulatorType:
            assert 0.0 <= ns.levels[nm] <= 1.0

    def test_modulation_factor_lr(self) -> None:
        """The learning-rate factor exceeds one."""
        ns = NeuromodulatorState()
        ns.levels[NeuromodulatorType.DOPAMINE] = 0.9
        assert ns.modulation_factor("lr") > 1.0

    def test_modulation_factor_default(self) -> None:
        """An unknown target has a factor of one."""
        ns = NeuromodulatorState()
        assert ns.modulation_factor("unknown") == 1.0

    def test_modulation_factor_tau(self) -> None:
        """The time-constant factor is one."""
        ns = NeuromodulatorState()
        # 0.8 + 0.4*(1 - ach=0.5) = 1.0
        assert ns.modulation_factor("tau") == pytest.approx(1.0)

    def test_modulation_factor_gain(self) -> None:
        """The gain factor is three quarters."""
        ns = NeuromodulatorState()
        # 0.5 + 0.5*ne=0.5 = 0.75
        assert ns.modulation_factor("gain") == pytest.approx(0.75)
