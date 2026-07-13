# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch biological learning autograd kernel

"""Custom Torch autograd transition shared by all plasticity rule types."""

from __future__ import annotations

from typing import Any

import torch

from .learning_validation import RULE_BCM, RULE_ELIGENT, RULE_REWARD_STDP, RULE_STDP


class _BiologicalAutogradFactory(torch.autograd.Function):
    """Evaluate biological transitions and their documented gradient surrogate."""

    @staticmethod
    def forward(
        ctx: Any,
        pre_spikes_f: torch.Tensor,
        post_spikes_f: torch.Tensor,
        rewards_f: torch.Tensor,
        weights: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
        eligibility: torch.Tensor,
        theta_m: torch.Tensor,
        act_avg: torch.Tensor,
        rule_params: torch.Tensor,
        rule_type: int,
        dt: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the next weight, trace, eligibility, and homeostatic state."""
        if ctx is not None:
            ctx.rule_type = rule_type
            ctx.dt = dt

        p_a_plus, p_a_minus, p_b_tau1, p_b_tau2, p_tau_e = rule_params
        next_weights = weights.clone()
        next_pre = pre_trace.clone()
        next_post = post_trace.clone()
        next_eligibility = eligibility.clone()
        next_theta = theta_m.clone()
        next_average = act_avg.clone()

        if rule_type == RULE_STDP:
            next_pre *= torch.exp(-dt / p_b_tau1)
            next_post *= torch.exp(-dt / p_b_tau2)
            potentiation = p_a_plus * next_pre * post_spikes_f
            depression = p_a_minus * next_post * pre_spikes_f
            next_weights = torch.clamp(next_weights + potentiation - depression, 0.0, 1.0)
            next_pre += pre_spikes_f
            next_post += post_spikes_f
        elif rule_type == RULE_REWARD_STDP:
            next_pre *= torch.exp(-dt / p_b_tau1)
            next_post *= torch.exp(-dt / p_b_tau2)
            potentiation = p_a_plus * next_pre * post_spikes_f
            depression = p_a_minus * next_post * pre_spikes_f
            next_eligibility += potentiation - depression
            next_eligibility *= torch.exp(-dt / p_tau_e)
            next_weights = torch.clamp(next_weights + next_eligibility * rewards_f, 0.0, 1.0)
            next_pre += pre_spikes_f
            next_post += post_spikes_f
        elif rule_type == RULE_ELIGENT:
            next_theta += p_a_minus * (post_spikes_f - p_a_plus) * dt
            next_eligibility += pre_spikes_f
            next_eligibility *= torch.exp(-dt / p_tau_e)
            delta = next_eligibility * rewards_f
            next_weights += delta
            next_average += delta
            active = next_average > 0.0
            next_weights = torch.where(
                active, next_weights / next_average, torch.zeros_like(next_weights)
            )
            next_average = torch.ones_like(next_average)
        elif rule_type == RULE_BCM:
            activity = post_spikes_f
            next_weights = torch.clamp(
                next_weights + p_a_plus * activity * (activity - next_theta) * pre_spikes_f * dt,
                0.0,
                1.0,
            )
            next_average += (activity - next_average) * (dt / p_b_tau1)
            next_theta += (next_average.square() - next_theta) * (dt / p_b_tau1)
            next_theta = torch.clamp(next_theta, min=0.01)

        if ctx is not None:
            ctx.save_for_backward(
                pre_spikes_f,
                post_spikes_f,
                rewards_f,
                next_weights,
                next_pre,
                next_post,
                next_eligibility,
                next_theta,
                next_average,
                rule_params,
            )
        return (
            next_weights,
            next_pre,
            next_post,
            next_eligibility,
            next_theta,
            next_average,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_weight: torch.Tensor,
        grad_pre_trace: torch.Tensor,
        grad_post_trace: torch.Tensor,
        grad_eligibility: torch.Tensor,
        grad_theta: torch.Tensor,
        grad_average: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Propagate the rule-specific surrogate gradient to public inputs."""
        del grad_pre_trace, grad_post_trace, grad_eligibility, grad_theta, grad_average
        (
            pre_spikes_f,
            post_spikes_f,
            rewards_f,
            _next_weights,
            next_pre,
            next_post,
            next_eligibility,
            next_theta,
            _next_average,
            rule_params,
        ) = ctx.saved_tensors
        p_a_plus, p_a_minus, _tau1, _tau2, _tau_e = rule_params
        grad_pre = grad_post = grad_reward = None

        if ctx.rule_type == RULE_STDP:
            grad_pre = grad_weight * (-p_a_minus * next_post)
            grad_post = grad_weight * (p_a_plus * next_pre)
        elif ctx.rule_type == RULE_REWARD_STDP:
            grad_pre = grad_weight * (-p_a_minus * next_post) * rewards_f
            grad_post = grad_weight * (p_a_plus * next_pre) * rewards_f
            grad_reward = grad_weight * next_eligibility
        elif ctx.rule_type == RULE_ELIGENT:
            grad_pre = grad_weight * (next_eligibility * rewards_f)
            grad_reward = grad_weight * next_eligibility
        elif ctx.rule_type == RULE_BCM:
            grad_pre = (
                grad_weight * p_a_plus * post_spikes_f * (post_spikes_f - next_theta) * ctx.dt
            )
            grad_post = (
                grad_weight * p_a_plus * (2 * post_spikes_f - next_theta) * pre_spikes_f * ctx.dt
            )
        return (
            grad_pre,
            grad_post,
            grad_reward,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
