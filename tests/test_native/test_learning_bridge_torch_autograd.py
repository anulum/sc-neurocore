# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning autograd tests

"""Rule-by-rule tests for the Torch biological transition and gradients."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch import TorchRuleLayer
from sc_neurocore._native.learning_torch_autograd import _BiologicalAutogradFactory
from sc_neurocore._native.learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    RULE_STDP,
)


def _state() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([0.0, 1.0, 1.0]),
        torch.tensor([0.2, -0.1, 0.3]),
        torch.full((3,), 0.5),
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.full((3,), 0.5),
        torch.zeros(3),
        torch.tensor([0.01, 0.005, 20.0, 20.0, 1.0]),
    )


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_direct_transition_without_autograd_context(rule_type: int) -> None:
    result = _BiologicalAutogradFactory.forward(None, *_state(), rule_type, 1.0)
    assert len(result) == 6
    assert all(tensor.shape == (3,) for tensor in result)
    assert all(bool(torch.all(torch.isfinite(tensor)).item()) for tensor in result)


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_autograd_routes_rule_specific_gradients(rule_type: int) -> None:
    layer = TorchRuleLayer(3, rule_type=rule_type, autograd=True)
    pre = torch.tensor([1.0, 0.0, 1.0], requires_grad=True)
    post = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
    reward = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
    result = layer.forward(pre, post, reward, dt=1.0)
    result.sum().backward()
    assert layer.weights.grad is not None
    assert pre.grad is not None
    if rule_type in (RULE_STDP, RULE_REWARD_STDP, RULE_BCM):
        assert post.grad is not None
    else:
        assert post.grad is None
    if rule_type in (RULE_REWARD_STDP, RULE_ELIGENT):
        assert reward.grad is not None
    else:
        assert reward.grad is None


def test_autograd_path_quantises_detached_trace_state() -> None:
    layer = TorchRuleLayer(
        3,
        rule_type=RULE_REWARD_STDP,
        autograd=True,
        trace_bits=3,
        eligibility_bits=3,
    )
    result = layer.forward(
        torch.tensor([1.0, 0.0, 1.0], requires_grad=True),
        torch.tensor([0.0, 1.0, 1.0], requires_grad=True),
        torch.ones(3, requires_grad=True),
    )
    assert result.grad_fn is not None
    assert layer.pre_trace.grad_fn is None
    assert layer.eligibility.grad_fn is None


def test_bcm_transition_tracks_activity_and_positive_threshold() -> None:
    layer = TorchRuleLayer(3, rule_type=RULE_BCM, autograd=False, param_b=1.0)
    before = layer.theta_m.clone()
    layer.forward(torch.ones(3), torch.ones(3), torch.zeros(3), dt=0.5)
    assert torch.all(layer.theta_m >= 0.01)
    assert not torch.equal(layer.theta_m, before)


def test_eligent_zero_average_branch_remains_finite() -> None:
    pre, post, reward, weight, pre_trace, post_trace, eligibility, theta, average, params = _state()
    average.zero_()
    reward.fill_(-100.0)
    result = _BiologicalAutogradFactory.forward(
        None,
        pre,
        post,
        reward,
        weight,
        pre_trace,
        post_trace,
        eligibility,
        theta,
        average,
        params,
        RULE_ELIGENT,
        1.0,
    )
    assert torch.all(torch.isfinite(result[0]))


def test_unknown_internal_rule_is_an_identity_transition_with_empty_gradients() -> None:
    state = _state()
    result = _BiologicalAutogradFactory.forward(None, *state, 99, 1.0)
    assert torch.equal(result[0], state[3])
    context = SimpleNamespace(saved_tensors=state, rule_type=99, dt=1.0)
    gradient = torch.ones(3)
    backward = _BiologicalAutogradFactory.backward(
        context,
        gradient,
        gradient,
        gradient,
        gradient,
        gradient,
        gradient,
    )
    assert backward[:3] == (None, None, None)
