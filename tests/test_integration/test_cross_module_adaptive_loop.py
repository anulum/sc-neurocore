# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveLoop from former test_cross_module.py

"""Focused suite: TestAdaptiveLoop from former test_cross_module.py."""

from __future__ import annotations

from cross_module_support import *  # noqa: F403


class TestAdaptiveLoop:
    """Verify Runtime → Optimizer closed loop."""

    def test_controller_creation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=64, mac_count=100)]
        ctrl = AdaptiveController(budget, layers)
        assert ctrl.current_report is None

    def test_no_drift_no_adaptation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        ctrl = AdaptiveController(budget, layers)

        # Feed uncorrelated bitstreams (no drift)
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.integers(0, 2, size=256).astype(np.float64)
            b = rng.integers(0, 2, size=256).astype(np.float64)
            event = ctrl.step(a, b)
        assert len(ctrl.adaptation_log) == 0

    def test_drift_triggers_adaptation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController, AdaptiveLoopConfig

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        config = AdaptiveLoopConfig(
            drift_threshold=0.05,
            reoptimize_cooldown_s=0.0,
            sa_max_iter=50,
        )
        ctrl = AdaptiveController(budget, layers, config)

        # Use 50%-density pattern so SCC is properly computed (not degenerate)
        rng = np.random.default_rng(42)
        pattern = rng.integers(0, 2, size=256).astype(np.float64)
        for _ in range(100):
            ctrl.step(pattern, pattern)  # identical ⇒ SCC=1.0 ⇒ drift

        assert len(ctrl.adaptation_log) >= 1

    def test_summary(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        ctrl = AdaptiveController(budget, layers)
        s = ctrl.summary()
        assert "AdaptiveController" in s

    def test_cooldown_suppresses_back_to_back_reoptimisation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import (
            AdaptiveController,
            AdaptiveLoopConfig,
        )

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        config = AdaptiveLoopConfig(
            drift_threshold=0.05,
            reoptimize_cooldown_s=100.0,
            sa_max_iter=50,
        )
        ctrl = AdaptiveController(budget, layers, config)
        rng = np.random.default_rng(42)
        pattern = rng.integers(0, 2, size=256).astype(np.float64)

        # Drive identical pairs until the first re-optimisation fires...
        first = None
        for _ in range(100):
            event = ctrl.step(pattern, pattern)
            if event is not None:
                first = event
                break
        assert first is not None
        # ...the very next step lands inside the 100 s cooldown window and is
        # suppressed, and the adaptation-rate property stays well defined.
        assert ctrl.step(pattern, pattern) is None
        assert 0.0 <= ctrl.adaptation_rate <= 1.0
