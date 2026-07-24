# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTorchRuleLayerReset from former test_autonomous_learning.py

"""Focused suite: TestTorchRuleLayerReset from former test_autonomous_learning.py."""

from __future__ import annotations

from autonomous_learning_support import *  # noqa: F403


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

    def test_scalar_mixed_precision_quantises_weights_and_traces(self):
        import torch

        layer = create_plasticity_layer(
            count=4,
            rule_type=RULE_STDP,
            backend="torch",
            autograd=False,
            weight=0.37,
            weight_bits=3,
            trace_bits=4,
            weight_clip=1.0,
            trace_clip=1.0,
        )

        pre = torch.tensor([1.0, 1.0, 1.0, 1.0])
        post = torch.tensor([0.0, 1.0, 0.0, 1.0])
        rewards = torch.zeros(4)
        for _ in range(8):
            layer.forward(pre, post, rewards, dt=1.0)

        w_step = 1.0 / ((2 ** (3 - 1)) - 1)  # 3-bit signed grid in [-1,1]
        t_step = 1.0 / ((2 ** (4 - 1)) - 1)  # 4-bit signed grid in [-1,1]
        scaled_w = layer.weights.detach() / w_step
        scaled_pre = layer.pre_trace.detach() / t_step
        scaled_post = layer.post_trace.detach() / t_step

        assert torch.allclose(scaled_w, torch.round(scaled_w), atol=1e-5)
        assert torch.allclose(scaled_pre, torch.round(scaled_pre), atol=1e-5)
        assert torch.allclose(scaled_post, torch.round(scaled_post), atol=1e-5)

    def test_per_synapse_weight_bits_accept_vector_spec(self):
        import torch

        bits = [2, 3, 4, 5]
        layer = create_plasticity_layer(
            count=4,
            rule_type=RULE_STDP,
            backend="torch",
            autograd=False,
            weight=0.49,
            weight_bits=bits,
            weight_clip=1.0,
        )

        pre = torch.ones(4)
        post = torch.tensor([1.0, 0.0, 1.0, 0.0])
        rewards = torch.zeros(4)
        layer.forward(pre, post, rewards, dt=1.0)

        weights = layer.weights.detach()
        for idx, bit_width in enumerate(bits):
            step = 1.0 / ((2 ** (bit_width - 1)) - 1)
            scaled = weights[idx] / step
            assert torch.allclose(scaled, torch.round(scaled), atol=1e-5)

    def test_rejects_malformed_weight_bits_vector_length(self):
        with pytest.raises(ValueError, match="weight_bits must be scalar or have length 4"):
            create_plasticity_layer(
                count=4,
                rule_type=RULE_STDP,
                backend="torch",
                autograd=False,
                weight_bits=[4, 4, 4],
            )
