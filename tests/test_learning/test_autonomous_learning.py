# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
import pytest

try:
    from sc_neurocore._native.learning_bridge import (
        is_available,
        RustPlasticityRule,
        RustEligentLearner,
        RULE_STDP,
        RULE_REWARD_STDP,
        RULE_BCM,
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
