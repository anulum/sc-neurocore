# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAutonomousLearning from former test_autonomous_learning.py

"""Focused suite: TestAutonomousLearning from former test_autonomous_learning.py."""

from __future__ import annotations

from autonomous_learning_support import *  # noqa: F403


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

    def test_online_o1_matches_python_reference_trace(self):
        from sc_neurocore.learning.online_o1 import OnlineO1Config, OnlineO1Synapse

        config = OnlineO1Config(
            weight_bits=8,
            trace_bits=6,
            reward_bits=4,
            learning_shift=3,
            trace_decay_shift=2,
        )
        events = [
            (True, False, 0),
            (False, True, 7),
            (False, False, 7),
            (False, False, 7),
            (False, False, -7),
            (True, False, 0),
            (False, True, -7),
        ]
        python_synapse = OnlineO1Synapse(config=config, initial_weight=0)
        rust_synapse = RustOnlineO1Synapse(
            weight_bits=8,
            trace_bits=6,
            reward_bits=4,
            learning_shift=3,
            trace_decay_shift=2,
            initial_weight=0,
        )

        for pre_spike, post_spike, reward in events:
            expected = python_synapse.step(
                pre_spike=pre_spike,
                post_spike=post_spike,
                reward=reward,
            )
            observed = rust_synapse.step(
                pre_spike=pre_spike,
                post_spike=post_spike,
                reward=reward,
            )
            assert observed.weight == expected.weight
            assert observed.pre_trace == expected.pre_trace
            assert observed.post_trace == expected.post_trace
            assert observed.eligibility == expected.eligibility

        assert rust_synapse.per_synapse_state_bits == config.per_synapse_state_bits
