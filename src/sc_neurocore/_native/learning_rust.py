# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Scalar autonomous-learning Rust wrappers

"""Memory-safe Python owners for scalar autonomous-learning Rust handles.

``ctypes.CDLL`` releases the Python GIL while these native calls execute.  Each
wrapper therefore owns exactly one handle and exposes explicit ``close`` and
context-manager boundaries in addition to best-effort finalisation.
"""

from __future__ import annotations

import ctypes as ct
from types import TracebackType
from typing import TYPE_CHECKING, Any

from . import learning_runtime as runtime
from .learning_validation import (
    MAX_I32,
    MAX_ONLINE_O1_REWARD_BITS,
    MAX_ONLINE_O1_SHIFT,
    MAX_ONLINE_O1_TRACE_BITS,
    MAX_ONLINE_O1_WEIGHT_BITS,
    MAX_U32,
    MIN_I32,
    as_bool_vector,
    as_float_vector,
    require_bool,
    require_finite_float,
    require_integral,
    require_integral_range,
    require_non_negative_float,
    require_non_negative_integral,
    require_positive_float,
    require_rule_type,
    require_unit_interval,
    saturate,
)

if TYPE_CHECKING:
    # ``typing.Self`` only exists on Python >= 3.11; import it lazily from
    # ``typing_extensions`` so the runtime import never executes on 3.10 while
    # type checkers still resolve the ``__enter__`` return annotation.
    from typing_extensions import Self


def _require_native() -> None:
    """Raise when the optional Rust autonomous-learning library is absent."""
    if not runtime.is_available():
        raise RuntimeError("libautonomous_learning is not available")


class _NativeOwner:
    """Common deterministic lifecycle for an opaque native handle."""

    _destroy_symbol = ""
    _ptr: Any

    def close(self) -> None:
        """Release the owned native handle; repeated calls are harmless."""
        pointer = getattr(self, "_ptr", None)
        if pointer:
            runtime.destroy_noexcept(self._destroy_symbol, pointer)
            self._ptr = None

    def __enter__(self) -> Self:
        """Return this live native owner for a ``with`` statement."""
        if not getattr(self, "_ptr", None):
            raise RuntimeError("native learning handle is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the handle when leaving a ``with`` statement."""
        self.close()

    def __del__(self) -> None:
        self.close()


class RustOnlineO1Synapse(_NativeOwner):
    """Own one bounded fixed-point online O(1) Rust learner."""

    __slots__ = ("_ptr",)
    _destroy_symbol = "destroy_online_o1_synapse"

    def __init__(
        self,
        *,
        weight_bits: int = 16,
        trace_bits: int = 12,
        reward_bits: int = 8,
        learning_shift: int = 4,
        trace_decay_shift: int = 4,
        initial_weight: int = 0,
    ) -> None:
        _require_native()
        create = runtime.require_symbol("create_online_o1_synapse")
        validated = (
            require_integral_range(
                name="weight_bits", value=weight_bits, lower=1, upper=MAX_ONLINE_O1_WEIGHT_BITS
            ),
            require_integral_range(
                name="trace_bits", value=trace_bits, lower=2, upper=MAX_ONLINE_O1_TRACE_BITS
            ),
            require_integral_range(
                name="reward_bits", value=reward_bits, lower=1, upper=MAX_ONLINE_O1_REWARD_BITS
            ),
            require_integral_range(
                name="learning_shift", value=learning_shift, lower=0, upper=MAX_ONLINE_O1_SHIFT
            ),
            require_integral_range(
                name="trace_decay_shift",
                value=trace_decay_shift,
                lower=0,
                upper=MAX_ONLINE_O1_SHIFT,
            ),
        )
        weight = min(
            require_non_negative_integral(name="initial_weight", value=initial_weight), MAX_U32
        )
        self._ptr = create(
            *(ct.c_uint8(value) for value in validated),
            ct.c_uint32(weight),
        )
        if not self._ptr:
            raise ValueError("invalid online O(1) fixed-point configuration")

    def step(
        self, *, pre_spike: bool, post_spike: bool, reward: int
    ) -> runtime.OnlineO1SnapshotFFI:
        """Advance one timestep and return the bounded fixed-point state."""
        snapshot: runtime.OnlineO1SnapshotFFI = runtime.require_symbol("step_online_o1_synapse")(
            self._ptr,
            require_bool(name="pre_spike", value=pre_spike),
            require_bool(name="post_spike", value=post_spike),
            ct.c_int32(saturate(require_integral(name="reward", value=reward), MIN_I32, MAX_I32)),
        )
        return snapshot

    @property
    def per_synapse_state_bits(self) -> int:
        """Return the Rust-reported fixed-point state footprint in bits."""
        return int(runtime.require_symbol("online_o1_per_synapse_state_bits")(self._ptr))


class RustPlasticityRule(_NativeOwner):
    """Own one scalar Rust STDP, R-STDP, BCM, or ELIGENT rule."""

    __slots__ = ("_ptr", "_rule_type", "_step_rule", "_get_rule_weight", "_reset_rule")
    _destroy_symbol = "destroy_rule"

    def __init__(
        self,
        rule_type: int = 1,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
    ) -> None:
        _require_native()
        self._rule_type = require_rule_type(rule_type)
        self._step_rule = runtime.require_symbol("step_rule")
        self._get_rule_weight = runtime.require_symbol("get_rule_weight")
        self._reset_rule = runtime.require_symbol("reset_rule")
        self._ptr = runtime.require_symbol("create_rule")(
            ct.c_uint32(self._rule_type),
            ct.c_float(require_unit_interval(name="weight", value=weight)),
            ct.c_float(require_non_negative_float(name="param_a", value=param_a)),
            ct.c_float(require_non_negative_float(name="param_b", value=param_b)),
        )
        if not self._ptr:
            raise RuntimeError("Rust plasticity rule construction failed")

    def step(
        self,
        pre_spike: bool,
        post_spike: bool,
        dt: float = 0.001,
        reward: float = 0.0,
    ) -> None:
        """Advance one scalar plasticity timestep through the Rust ABI."""
        self._step_rule(
            self._ptr,
            require_bool(name="pre_spike", value=pre_spike),
            require_bool(name="post_spike", value=post_spike),
            ct.c_float(require_finite_float(name="reward", value=reward)),
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )

    def step_batched(
        self,
        pre_spikes: object,
        post_spikes: object,
        rewards: object,
        dt: float = 0.001,
    ) -> None:
        """Advance equally sized vectors in one native boundary crossing."""
        pre = as_bool_vector(pre_spikes, name="pre_spikes")
        post = as_bool_vector(post_spikes, name="post_spikes", length=pre.size)
        reward = as_float_vector(rewards, name="rewards", length=pre.size)
        runtime.require_symbol("step_rule_batched")(
            self._ptr,
            pre.ctypes.data_as(ct.POINTER(ct.c_bool)),
            post.ctypes.data_as(ct.POINTER(ct.c_bool)),
            reward.ctypes.data_as(ct.POINTER(ct.c_float)),
            ct.c_size_t(pre.size),
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )

    @property
    def weight(self) -> float:
        """Return the current Rust-managed rule weight."""
        return float(self._get_rule_weight(self._ptr))

    def reset(self) -> None:
        """Clear rule traces while retaining the learned weight."""
        self._reset_rule(self._ptr)


class RustEligentLearner(_NativeOwner):
    """Own one backward-compatible scalar Rust ELIGENT learner."""

    __slots__ = ("_ptr",)
    _destroy_symbol = "destroy_learner"

    def __init__(
        self,
        threshold: float = 1.0,
        target_rate: float = 0.1,
        weight: float = 0.5,
    ) -> None:
        _require_native()
        self._ptr = runtime.require_symbol("create_learner")(
            ct.c_float(require_positive_float(name="threshold", value=threshold)),
            ct.c_float(require_non_negative_float(name="target_rate", value=target_rate)),
            ct.c_float(require_unit_interval(name="weight", value=weight)),
        )
        if not self._ptr:
            raise RuntimeError("Rust ELIGENT learner construction failed")

    def step(
        self,
        fired: bool,
        pre_spike: bool,
        global_reward: float = 0.0,
        dt: float = 0.001,
    ) -> None:
        """Advance one scalar ELIGENT timestep through the Rust ABI."""
        runtime.require_symbol("step_learner")(
            self._ptr,
            require_bool(name="fired", value=fired),
            require_bool(name="pre_spike", value=pre_spike),
            ct.c_float(require_finite_float(name="global_reward", value=global_reward)),
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )

    def step_batched(
        self,
        fired_slice: object,
        pre_spikes: object,
        rewards: object,
        dt: float = 0.001,
    ) -> None:
        """Advance equally sized ELIGENT vectors through one native call."""
        fired = as_bool_vector(fired_slice, name="fired_slice")
        pre = as_bool_vector(pre_spikes, name="pre_spikes", length=fired.size)
        reward = as_float_vector(rewards, name="rewards", length=fired.size)
        runtime.require_symbol("step_learner_batched")(
            self._ptr,
            fired.ctypes.data_as(ct.POINTER(ct.c_bool)),
            pre.ctypes.data_as(ct.POINTER(ct.c_bool)),
            reward.ctypes.data_as(ct.POINTER(ct.c_float)),
            ct.c_size_t(fired.size),
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )
