# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning layer state and forward contracts

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch import TorchRuleLayer
from sc_neurocore._native.learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    RULE_STDP,
)


def test_torch_layer_initialises_all_state_and_precision() -> None:
    layer = TorchRuleLayer(
        3,
        rule_type=RULE_ELIGENT,
        weight=0.4,
        mixed_precision_bits=4,
        trace_bits=[3, 4, 5],
    )
    assert layer.weights.tolist() == pytest.approx([0.4] * 3)
    assert layer.theta_m.tolist() == pytest.approx([1.0] * 3)
    assert layer.act_avg.tolist() == pytest.approx([0.4] * 3)
    assert torch.equal(layer._weight_bits, torch.tensor([4, 4, 4]))
    assert torch.equal(layer._trace_bits, torch.tensor([3, 4, 5]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"rule_type": 9},
        {"weight": -1.0},
        {"autograd": 1},
        {"count": 3, "unexpected": 1},
    ],
)
def test_torch_layer_rejects_invalid_constructor_options(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor = cast(Callable[..., TorchRuleLayer], TorchRuleLayer)
        constructor(**kwargs)


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_non_autograd_forward_and_reset(rule_type: int) -> None:
    layer = TorchRuleLayer(3, rule_type=rule_type, autograd=False, weight_bits=4)
    pre = torch.tensor([1.0, 0.0, 1.0])
    post = torch.tensor([0.0, 1.0, 1.0])
    rewards = torch.tensor([0.2, -0.1, 0.3])
    result = layer.forward(pre, post, rewards, dt=1.0)
    assert result is layer.weights and torch.all(torch.isfinite(result))
    layer.reset()
    if rule_type == RULE_STDP:
        assert torch.count_nonzero(layer.pre_trace) == 0
    elif rule_type == RULE_REWARD_STDP:
        assert torch.count_nonzero(layer.eligibility) == 0
    elif rule_type == RULE_BCM:
        assert torch.all(layer.theta_m == 0.5)
    else:
        assert torch.count_nonzero(layer.eligibility) == 0


def test_forward_warning_validation_numpy_step_and_state() -> None:
    layer = TorchRuleLayer(3, rule_type=RULE_REWARD_STDP)
    with pytest.warns(UserWarning, match="expects 'rewards'"):
        layer.forward(torch.ones(3), torch.zeros(3), rewards=None)
    layer.step(np.ones(3), np.zeros(3), np.array([0.1, 0.2, 0.3]))
    state = layer.get_state_dict()
    clone = TorchRuleLayer(3, rule_type=RULE_REWARD_STDP)
    clone.load_state_dict(state)
    assert clone.get_weights() == pytest.approx(layer.get_weights())
    assert clone.get_weights() is not layer.get_weights()


def test_forward_without_reward_is_quiet_for_unsupervised_rule() -> None:
    layer = TorchRuleLayer(3, rule_type=RULE_STDP)
    with warnings.catch_warnings(record=True) as captured:
        layer.forward(torch.ones(3), torch.zeros(3), rewards=None)
    assert not captured


def test_forward_rejects_shape_values_and_timestep() -> None:
    layer = TorchRuleLayer(3)
    with pytest.raises(ValueError, match="shape"):
        layer.forward(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="values in"):
        layer.forward(torch.ones(3) * 2, torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        layer.forward(torch.ones(3), torch.ones(3), torch.tensor([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="dt"):
        layer.forward(torch.ones(3), torch.ones(3), dt=0.0)
