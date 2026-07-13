# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WGPU autonomous-learning wrapper

"""Length-checked Python ownership for the Rust WGPU plasticity layer."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes as ct
from typing import Any

import numpy as np

from . import learning_runtime as runtime
from .learning_rust import _NativeOwner, _require_native
from .learning_validation import (
    as_float_vector,
    as_probability_vector,
    require_count,
    require_non_negative_float,
    require_positive_float,
    require_rule_type,
    require_u32_seed,
    require_unit_interval,
)


class RustWgpuRuleLayer(_NativeOwner):
    """Own one explicit WGPU rule layer with safe host-buffer boundaries."""

    __slots__ = ("_ptr", "_count", "_rule_type")
    _destroy_symbol = "free_wgpu_layer"

    def __init__(
        self,
        count: int,
        rule_type: int = 1,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
        tau_e: float = 20.0,
        target_sum_weights: float = 1.0,
        **kwargs: Any,
    ) -> None:
        _require_native()
        self._count = require_count(count)
        self._rule_type = require_rule_type(rule_type)
        initial_weight = require_unit_interval(name="weight", value=weight)
        a_plus = require_non_negative_float(name="param_a", value=param_a)
        a_minus = require_non_negative_float(name="param_b", value=param_b)
        tau_plus = require_positive_float(name="tau_plus", value=kwargs.get("tau_plus", 20.0))
        tau_minus = require_positive_float(name="tau_minus", value=kwargs.get("tau_minus", 20.0))
        tau_eligibility = require_positive_float(name="tau_e", value=tau_e)
        target_sum = require_non_negative_float(name="target_sum_weights", value=target_sum_weights)
        common = (
            ct.c_size_t(self._count),
            ct.c_uint32(self._rule_type),
        )
        parameters = (
            ct.c_float(a_plus),
            ct.c_float(a_minus),
            ct.c_float(tau_plus),
            ct.c_float(tau_minus),
            ct.c_float(tau_eligibility),
            ct.c_float(target_sum),
        )
        if runtime.has_symbol("create_wgpu_layer_with_weight"):
            self._ptr = runtime.require_symbol("create_wgpu_layer_with_weight")(
                *common, ct.c_float(initial_weight), *parameters
            )
        else:
            if initial_weight != 0.5:
                raise RuntimeError(
                    "loaded autonomous-learning library cannot set a WGPU initial weight"
                )
            self._ptr = runtime.require_symbol("create_wgpu_layer")(*common, *parameters)
        if not self._ptr:
            raise RuntimeError(
                "WGPU initialization failed; verify Vulkan, Metal, WebGPU, or DX12 support"
            )
        configured_seed = runtime.deterministic_seed()
        if configured_seed is not None:
            self._set_seed(configured_seed)

    def _set_seed(self, seed: object) -> None:
        """Apply one validated deterministic seed to the native layer."""
        runtime.require_symbol("set_wgpu_layer_seed")(
            self._ptr, ct.c_uint32(require_u32_seed(name="seed", value=seed))
        )

    def step(
        self,
        pre_spikes: object,
        post_spikes: object,
        rewards: object | None = None,
        dt: float = 0.001,
    ) -> None:
        """Advance one exactly sized WGPU probability batch."""
        pre = as_probability_vector(pre_spikes, name="pre_spikes", length=self._count)
        post = as_probability_vector(post_spikes, name="post_spikes", length=self._count)
        reward = (
            None
            if rewards is None
            else as_float_vector(rewards, name="rewards", length=self._count)
        )
        reward_pointer = None if reward is None else reward.ctypes.data_as(ct.POINTER(ct.c_float))
        runtime.require_symbol("step_wgpu_layer")(
            self._ptr,
            pre.ctypes.data_as(ct.POINTER(ct.c_float)),
            post.ctypes.data_as(ct.POINTER(ct.c_float)),
            reward_pointer,
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )

    def step_analog(
        self,
        pre_probs: object,
        post_probs: object,
        rewards: object,
        dt: float = 0.001,
        seed: int | None = None,
    ) -> None:
        """Advance probability vectors after optionally reseeding WGPU."""
        configured = runtime.deterministic_seed()
        if configured is not None:
            self._set_seed(configured)
        elif seed is not None:
            self._set_seed(seed)
        self.step(pre_probs, post_probs, rewards, dt)

    def get_weights(self) -> np.ndarray[Any, Any]:
        """Return a detached copy of all WGPU-managed weights."""
        result = np.empty(self._count, dtype=np.float32)
        runtime.require_symbol("get_wgpu_weights")(
            self._ptr, result.ctypes.data_as(ct.POINTER(ct.c_float))
        )
        return result

    def get_state_dict(self) -> dict[str, Any]:
        """Return the portable WGPU weight state."""
        return {"weights": self.get_weights()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore weights through the length-aware WGPU Rust ABI."""
        if "weights" not in state_dict:
            raise ValueError("WGPU state is missing weights")
        weights = as_float_vector(state_dict["weights"], name="weights", length=self._count)
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("weights must be in [0, 1]")
        setter = runtime.require_symbol("set_wgpu_weights")
        if not setter(
            self._ptr,
            weights.ctypes.data_as(ct.POINTER(ct.c_float)),
            ct.c_size_t(self._count),
        ):
            raise RuntimeError("WGPU weight restoration failed")

    def reset(self) -> None:
        """Clear WGPU plasticity traces while preserving weights."""
        runtime.require_symbol("reset_wgpu_layer")(self._ptr)
