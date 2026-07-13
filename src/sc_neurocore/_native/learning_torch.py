# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch autonomous-learning layer

"""Torch implementation of the four autonomous plasticity rule families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
import warnings

import numpy as np
import torch
import torch.nn as nn

from .learning_torch_autograd import _BiologicalAutogradFactory
from .learning_torch_precision import normalise_bit_spec, normalise_clip, quantise_tensor
from .learning_torch_support import (
    KNOWN_KWARGS,
    PRECISION_NAMES,
    rule_parameters,
    validate_input,
)
from .learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    RULE_STDP,
    require_bool,
    require_count,
    require_positive_float,
    require_rule_type,
    require_unit_interval,
)


class TorchRuleLayer(nn.Module):
    """Execute biological plasticity with optional surrogate autograd."""

    rule_params: torch.Tensor
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    eligibility: torch.Tensor
    theta_m: torch.Tensor
    act_avg: torch.Tensor

    def __init__(
        self,
        count: int,
        rule_type: int = RULE_STDP,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
        autograd: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        unknown = sorted(set(kwargs) - KNOWN_KWARGS)
        if unknown:
            raise TypeError(f"unknown Torch learning options: {', '.join(unknown)}")
        self.count = require_count(count)
        self.rule_type = require_rule_type(rule_type)
        self.autograd = require_bool(name="autograd", value=autograd)
        initial_weight = require_unit_interval(name="weight", value=weight)
        parameters = rule_parameters(self.rule_type, param_a, param_b, kwargs)
        self.register_buffer("rule_params", torch.tensor(parameters, dtype=torch.float32))
        self.weights = nn.Parameter(
            torch.full((self.count,), initial_weight, dtype=torch.float32),
            requires_grad=self.autograd,
        )
        self.register_buffer("pre_trace", torch.zeros(self.count, dtype=torch.float32))
        self.register_buffer("post_trace", torch.zeros(self.count, dtype=torch.float32))
        self.register_buffer("eligibility", torch.zeros(self.count, dtype=torch.float32))
        theta = 1.0 if self.rule_type == RULE_ELIGENT else 0.5
        average = initial_weight if self.rule_type == RULE_ELIGENT else 0.0
        self.register_buffer("theta_m", torch.full((self.count,), theta, dtype=torch.float32))
        self.register_buffer("act_avg", torch.full((self.count,), average, dtype=torch.float32))
        default_bits = kwargs.get("mixed_precision_bits")
        for name in PRECISION_NAMES:
            bits = normalise_bit_spec(
                kwargs.get(f"{name}_bits", default_bits),
                count=self.count,
                device=self.weights.device,
                field=f"{name}_bits",
            )
            setattr(self, f"_{name}_bits", bits)
            setattr(
                self,
                f"_{name}_clip",
                normalise_clip(kwargs.get(f"{name}_clip", 1.0), field=f"{name}_clip"),
            )

    def _quantise(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """Apply the configured quantiser for one state family."""
        bits: torch.Tensor | None = getattr(self, f"_{name}_bits")
        clip: float = getattr(self, f"_{name}_clip")
        return quantise_tensor(values, bits, clip)

    def _apply_precision_constraints(self) -> None:
        """Quantise every mutable weight and trace tensor in place."""
        self.weights.copy_(self._quantise(self.weights, "weight"))
        self.pre_trace.copy_(self._quantise(self.pre_trace, "trace"))
        self.post_trace.copy_(self._quantise(self.post_trace, "trace"))
        self.eligibility.copy_(self._quantise(self.eligibility, "eligibility"))
        self.theta_m.copy_(self._quantise(self.theta_m, "theta"))
        self.act_avg.copy_(self._quantise(self.act_avg, "act_avg"))

    def reset(self) -> None:
        """Clear only the mutable traces defined by the selected rule."""
        with torch.no_grad():
            if self.rule_type == RULE_STDP:
                self.pre_trace.zero_()
                self.post_trace.zero_()
            elif self.rule_type == RULE_REWARD_STDP:
                self.pre_trace.zero_()
                self.post_trace.zero_()
                self.eligibility.zero_()
            elif self.rule_type == RULE_BCM:
                self.act_avg.zero_()
                self.theta_m.fill_(0.5)
            else:
                self.eligibility.zero_()

    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        rewards: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Advance one vector timestep and return the resulting weights."""
        timestep = require_positive_float(name="dt", value=dt)
        pre = validate_input(
            pre_spikes,
            name="pre_spikes",
            count=self.count,
            device=self.weights.device,
            dtype=self.weights.dtype,
            probability=True,
        )
        post = validate_input(
            post_spikes,
            name="post_spikes",
            count=self.count,
            device=self.weights.device,
            dtype=self.weights.dtype,
            probability=True,
        )
        if rewards is None:
            if self.rule_type in (RULE_ELIGENT, RULE_REWARD_STDP):
                warnings.warn(
                    f"TorchRuleLayer rule_type {self.rule_type} expects 'rewards'; using zeros",
                    UserWarning,
                    stacklevel=2,
                )
            reward = torch.zeros_like(pre)
        else:
            reward = validate_input(
                rewards,
                name="rewards",
                count=self.count,
                device=self.weights.device,
                dtype=self.weights.dtype,
                probability=False,
            )
        if self.autograd:
            state = cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                _BiologicalAutogradFactory.apply(
                    pre,
                    post,
                    reward,
                    self.weights,
                    self.pre_trace,
                    self.post_trace,
                    self.eligibility,
                    self.theta_m,
                    self.act_avg,
                    self.rule_params,
                    self.rule_type,
                    timestep,
                ),
            )
        else:
            state = _BiologicalAutogradFactory.forward(
                None,
                pre,
                post,
                reward,
                self.weights,
                self.pre_trace,
                self.post_trace,
                self.eligibility,
                self.theta_m,
                self.act_avg,
                self.rule_params,
                self.rule_type,
                timestep,
            )
        next_weight, next_pre, next_post, next_eligibility, next_theta, next_average = state
        if self.autograd:
            self.pre_trace = self._quantise(next_pre.detach().clone(), "trace")
            self.post_trace = self._quantise(next_post.detach().clone(), "trace")
            self.eligibility = self._quantise(next_eligibility.detach().clone(), "eligibility")
            self.theta_m = self._quantise(next_theta.detach().clone(), "theta")
            self.act_avg = self._quantise(next_average.detach().clone(), "act_avg")
            return next_weight
        with torch.no_grad():
            for target, source in zip(
                (
                    self.weights,
                    self.pre_trace,
                    self.post_trace,
                    self.eligibility,
                    self.theta_m,
                    self.act_avg,
                ),
                state,
                strict=True,
            ):
                target.copy_(source)
            self._apply_precision_constraints()
        return self.weights

    def step(
        self,
        pre_spikes: torch.Tensor | np.ndarray[Any, Any],
        post_spikes: torch.Tensor | np.ndarray[Any, Any],
        rewards: torch.Tensor | np.ndarray[Any, Any],
        dt: float = 1.0,
    ) -> None:
        """Compatibility wrapper accepting Torch or NumPy vectors."""
        device = self.weights.device
        pre = (
            pre_spikes
            if isinstance(pre_spikes, torch.Tensor)
            else torch.as_tensor(pre_spikes, device=device)
        )
        post = (
            post_spikes
            if isinstance(post_spikes, torch.Tensor)
            else torch.as_tensor(post_spikes, device=device)
        )
        reward = (
            rewards
            if isinstance(rewards, torch.Tensor)
            else torch.as_tensor(rewards, device=device)
        )
        self.forward(pre, post, reward, dt)

    def get_state_dict(self) -> dict[str, Any]:
        """Return the standard Torch state as a plain dictionary."""
        return dict(self.state_dict())

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Restore a standard Torch state dictionary."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_weights(self) -> np.ndarray[Any, Any]:
        """Return a detached CPU copy of every weight."""
        result: np.ndarray[Any, Any] = self.weights.detach().cpu().numpy().copy()
        return result
