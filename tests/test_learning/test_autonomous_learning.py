# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
import numpy as np
import pytest

try:
    from sc_neurocore._native.learning_bridge import (
        is_available,
        RustPlasticityRule,
        RustEligentLearner,
        RustRuleLayer,
        RULE_ELIGENT,
        RULE_STDP,
        RULE_REWARD_STDP,
        RULE_BCM,
        create_plasticity_layer,
    )

    FFI_AVAILABLE = is_available()
except ImportError:
    FFI_AVAILABLE = False


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust autonomous_learning shared library not built.")
class TestAutonomousLearning:
    def test_stdp_rule_potentiation(self):
        rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
        initial_w = rule.weight

        # Pre-before-post (LTP)
        rule.step(True, False)
        rule.step(False, True)

        assert rule.weight > initial_w

    def test_stdp_rule_depression(self):
        rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
        initial_w = rule.weight

        # Post-before-pre (LTD)
        rule.step(False, True)
        rule.step(True, False)

        assert rule.weight < initial_w

    def test_reward_stdp_needs_reward(self):
        rule = RustPlasticityRule(rule_type=RULE_REWARD_STDP, weight=0.5, param_a=0.1, param_b=0.95)
        initial_w = rule.weight

        rule.step(True, False, reward=0.0)
        rule.step(False, True, reward=0.0)

        assert abs(rule.weight - initial_w) < 1e-6

        rule.step(True, False, reward=0.0)
        rule.step(False, True, reward=1.0)

        assert rule.weight != initial_w

    def test_bcm_rule(self):
        rule = RustPlasticityRule(rule_type=RULE_BCM, weight=0.5, param_a=0.01, param_b=10.0)
        initial_w = rule.weight
        rule.step(True, True)
        assert rule.weight != initial_w

    def test_eligent_learner(self):
        learner = RustEligentLearner(threshold=1.0, target_rate=0.1, weight=0.5)
        learner.step(fired=True, pre_spike=True, global_reward=1.0)
        # Should not crash and execute C-FFI correctly


# Mock imports for MetaPlasticity integration testing
from sc_neurocore.meta_plasticity.meta_plasticity import MetaPlasticityEngine, EngineConfig


class TestMetaPlasticityIntegration:
    def test_meta_plasticity_engine_step(self):
        engine = MetaPlasticityEngine(config=EngineConfig(enable_evolution=False))
        metrics = {
            "novelty": 0.8,
            "surprise": 0.1,
            "gci": 0.7,
            "gci_std": 0.05,
            "mean_rate_hz": 4.5,
        }

        res = engine.step(metrics)
        assert res["step"] == 1
        assert "current_rules" in res
        assert engine.neuromod.levels is not None


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


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust autonomous_learning shared library not built.")
class TestTorchRuleLayerReset:
    """``TorchRuleLayer.reset()`` must zero traces while preserving weights.

    Mirrors the per-rule Rust contract and matches each rule's reset
    scope: STDP → pre/post traces; REWARD_STDP → pre/post + eligibility;
    BCM → act_avg (zero) + theta_m (0.5); ELIGENT → eligibility.
    """

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def _drive(self, rule_type: int):
        import torch

        layer = create_plasticity_layer(count=4, rule_type=rule_type, backend="torch")
        pre = torch.ones(4)
        post = torch.ones(4)
        rewards = torch.full((4,), 0.1)
        for _ in range(10):
            layer.forward(pre, post, rewards, dt=1.0)
        return layer

    @pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
    def test_reset_preserves_weights(self, rule_type):
        layer = self._drive(rule_type)
        before = layer.get_weights().copy()
        layer.reset()
        assert np.allclose(layer.get_weights(), before)

    def test_stdp_reset_zeros_pre_and_post_traces(self):
        import torch

        layer = self._drive(RULE_STDP)
        assert layer.pre_trace.abs().sum().item() > 0.0  # precondition
        layer.reset()
        assert torch.all(layer.pre_trace == 0.0)
        assert torch.all(layer.post_trace == 0.0)

    def test_reward_stdp_reset_zeros_eligibility(self):
        import torch

        layer = self._drive(RULE_REWARD_STDP)
        layer.reset()
        assert torch.all(layer.eligibility == 0.0)

    def test_bcm_reset_restores_theta_m_to_half(self):
        import torch

        layer = self._drive(RULE_BCM)
        layer.reset()
        assert torch.all(layer.act_avg == 0.0)
        assert torch.allclose(layer.theta_m, torch.full_like(layer.theta_m, 0.5))

    def test_eligent_reset_zeros_eligibility_only(self):
        import torch

        layer = self._drive(RULE_ELIGENT)
        layer.reset()
        assert torch.all(layer.eligibility == 0.0)
