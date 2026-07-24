# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustRuleLayerReset from former test_autonomous_learning.py

"""Focused suite: TestRustRuleLayerReset from former test_autonomous_learning.py."""

from __future__ import annotations

from autonomous_learning_support import *  # noqa: F403


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust autonomous_learning shared library not built.")
class TestRustRuleLayerReset:
    """``RustRuleLayer.reset()`` must zero traces while preserving learned weights.

    Mirrors the per-rule ``PlasticityRule::reset`` contract in the Rust
    crate (``lib.rs``): STDP clears pre/post traces; REWARD_STDP also
    clears eligibility; BCM resets act_avg + theta_m; ELIGENT clears the
    eligibility trace. In every case the learned ``weight`` is preserved —
    the layer keeps its identity across a reset.
    """

    def _drive_and_capture(self, rule_type: int):
        layer = RustRuleLayer(count=4, rule_type=rule_type, weight=0.5, param_a=0.1, param_b=0.05)
        pre = np.ones(4, dtype=bool)
        post = np.ones(4, dtype=bool)
        rewards = np.full(4, 0.1, dtype=np.float32)
        for _ in range(20):
            layer.step(pre, post, rewards, dt=0.001)
        weights_before = layer.get_weights().copy()
        layer.reset()
        return layer, weights_before

    @pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
    def test_reset_preserves_weights(self, rule_type):
        layer, weights_before = self._drive_and_capture(rule_type)
        assert np.allclose(layer.get_weights(), weights_before)

    def test_reset_is_idempotent(self):
        layer, _ = self._drive_and_capture(RULE_STDP)
        snap = layer.get_weights().copy()
        layer.reset()
        layer.reset()
        assert np.allclose(layer.get_weights(), snap)

    def test_reset_followed_by_step_still_works(self):
        layer, _ = self._drive_and_capture(RULE_REWARD_STDP)
        # After reset we must be able to continue stepping without crash
        # and the engine must keep producing finite weights.
        pre = np.ones(4, dtype=bool)
        post = np.zeros(4, dtype=bool)
        rewards = np.full(4, 0.2, dtype=np.float32)
        for _ in range(5):
            layer.step(pre, post, rewards, dt=0.001)
        weights = layer.get_weights()
        assert np.all(np.isfinite(weights))
