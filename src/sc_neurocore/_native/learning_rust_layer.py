# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rayon autonomous-learning layer wrapper

"""Memory-safe owner for the batched Rust plasticity layer."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np

from . import learning_runtime as runtime
from .learning_rust import _NativeOwner, _require_native
from .learning_validation import (
    MAX_U64,
    as_bool_vector,
    as_float_vector,
    as_probability_vector,
    require_count,
    require_non_negative_float,
    require_positive_float,
    require_rule_type,
    require_u64_seed,
    require_unit_interval,
)

_STATE_FIELDS = ("count", "rule_type", "weight", "param_a", "param_b", "mem_buffer")


def _validated_state(state: Mapping[str, object]) -> tuple[int, int, float, float, float, bytes]:
    """Validate Python metadata and copy the opaque Rust state payload."""
    missing = [field for field in _STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"rule-layer state is missing fields: {', '.join(missing)}")
    payload = state["mem_buffer"]
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("mem_buffer must be bytes-like")
    data = bytes(payload)
    if not data:
        raise ValueError("mem_buffer must not be empty")
    return (
        require_count(state["count"]),
        require_rule_type(state["rule_type"]),
        require_unit_interval(name="weight", value=state["weight"]),
        require_non_negative_float(name="param_a", value=state["param_a"]),
        require_non_negative_float(name="param_b", value=state["param_b"]),
        data,
    )


class RustRuleLayer(_NativeOwner):
    """Own a parallel Rust rule layer with length-checked array boundaries."""

    __slots__ = (
        "_ptr",
        "_count",
        "_rule_type",
        "_weight",
        "_param_a",
        "_param_b",
        "_analog_seed_counter",
    )
    _destroy_symbol = "destroy_rule_layer"

    def __init__(
        self,
        count: int,
        rule_type: int = 1,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
    ) -> None:
        _require_native()
        values = (
            require_count(count),
            require_rule_type(rule_type),
            require_unit_interval(name="weight", value=weight),
            require_non_negative_float(name="param_a", value=param_a),
            require_non_negative_float(name="param_b", value=param_b),
        )
        pointer = self._create_handle(*values)
        self._install(pointer, *values)

    @staticmethod
    def _create_handle(
        count: int, rule_type: int, weight: float, param_a: float, param_b: float
    ) -> Any:
        """Construct one native layer or fail before ownership transfer."""
        pointer = runtime.require_symbol("create_rule_layer")(
            ct.c_size_t(count),
            ct.c_uint32(rule_type),
            ct.c_float(weight),
            ct.c_float(param_a),
            ct.c_float(param_b),
        )
        if not pointer:
            raise RuntimeError("Rust rule-layer construction failed")
        return pointer

    def _install(
        self,
        pointer: Any,
        count: int,
        rule_type: int,
        weight: float,
        param_a: float,
        param_b: float,
    ) -> None:
        """Install a fully constructed handle and its immutable metadata."""
        self._ptr = pointer
        self._count = count
        self._rule_type = rule_type
        self._weight = weight
        self._param_a = param_a
        self._param_b = param_b
        self._analog_seed_counter = 42

    @staticmethod
    def _restore_payload(pointer: Any, payload: bytes) -> None:
        """Restore only through the length-aware Rust parser."""
        restore = runtime.require_symbol("set_rule_layer_state_mem_checked")
        buffer = (ct.c_uint8 * len(payload)).from_buffer_copy(payload)
        if not restore(pointer, buffer, ct.c_size_t(len(payload))):
            raise ValueError("invalid or incompatible Rust rule-layer state")

    def __getstate__(self) -> dict[str, Any]:
        """Return validated metadata and Rust-owned serialization bytes."""
        size = int(runtime.require_symbol("get_rule_layer_state_size")(self._ptr))
        if size <= 0:
            raise RuntimeError("Rust rule-layer state size query failed")
        buffer = (ct.c_uint8 * size)()
        if not runtime.require_symbol("get_rule_layer_state_mem")(self._ptr, buffer):
            raise RuntimeError("Rust rule-layer state serialization failed")
        return {
            "count": self._count,
            "rule_type": self._rule_type,
            "weight": self._weight,
            "param_a": self._param_a,
            "param_b": self._param_b,
            "mem_buffer": bytes(buffer),
        }

    def get_state_dict(self) -> dict[str, Any]:
        """Return a Python state dictionary containing Rust serialization."""
        return self.__getstate__()

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Atomically replace this layer from a validated serialized state."""
        if not runtime.is_available() and not runtime._load_native_library():
            raise RuntimeError("libautonomous_learning is not available")
        count, rule_type, weight, param_a, param_b, payload = _validated_state(state)
        pointer = self._create_handle(count, rule_type, weight, param_a, param_b)
        try:
            self._restore_payload(pointer, payload)
        except BaseException:
            runtime.destroy_noexcept(self._destroy_symbol, pointer)
            raise
        old_pointer = getattr(self, "_ptr", None)
        self._install(pointer, count, rule_type, weight, param_a, param_b)
        runtime.destroy_noexcept(self._destroy_symbol, old_pointer)

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Atomically restore this layer from a Python state dictionary."""
        self.__setstate__(state_dict)

    def step(
        self,
        pre_spikes: object,
        post_spikes: object,
        rewards: object,
        dt: float = 0.001,
    ) -> None:
        """Process one exactly sized spatial batch on Rayon threads."""
        pre = as_bool_vector(pre_spikes, name="pre_spikes", length=self._count)
        post = as_bool_vector(post_spikes, name="post_spikes", length=self._count)
        reward = as_float_vector(rewards, name="rewards", length=self._count)
        runtime.require_symbol("step_rule_layer")(
            self._ptr,
            pre.ctypes.data_as(ct.POINTER(ct.c_bool)),
            post.ctypes.data_as(ct.POINTER(ct.c_bool)),
            reward.ctypes.data_as(ct.POINTER(ct.c_float)),
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
        """Sample probability vectors natively using an explicit safe seed."""
        configured = runtime.deterministic_seed()
        if configured is not None:
            actual_seed = configured
        elif seed is not None:
            actual_seed = require_u64_seed(name="seed", value=seed)
        else:
            actual_seed = self._analog_seed_counter
            self._analog_seed_counter = (actual_seed + 1) & MAX_U64
        pre = as_probability_vector(pre_probs, name="pre_probs", length=self._count)
        post = as_probability_vector(post_probs, name="post_probs", length=self._count)
        reward = as_float_vector(rewards, name="rewards", length=self._count)
        runtime.require_symbol("step_rule_layer_analog")(
            self._ptr,
            pre.ctypes.data_as(ct.POINTER(ct.c_float)),
            post.ctypes.data_as(ct.POINTER(ct.c_float)),
            reward.ctypes.data_as(ct.POINTER(ct.c_float)),
            ct.c_uint64(actual_seed),
            ct.c_float(require_positive_float(name="dt", value=dt)),
        )

    def get_weights(self) -> np.ndarray[Any, Any]:
        """Return a detached copy of every native rule weight."""
        result = np.empty(self._count, dtype=np.float32)
        runtime.require_symbol("get_rule_layer_weights")(
            self._ptr, result.ctypes.data_as(ct.POINTER(ct.c_float))
        )
        return result

    def save(self, path: str) -> bool:
        """Write the checked in-memory state format to ``path``."""
        Path(path).expanduser().resolve().write_bytes(self.__getstate__()["mem_buffer"])
        return True

    def load(self, path: str) -> bool:
        """Read and atomically restore a checked state payload from ``path``."""
        state = self.__getstate__()
        state["mem_buffer"] = Path(path).expanduser().resolve().read_bytes()
        self.__setstate__(state)
        return True

    def reset(self) -> None:
        """Clear every rule trace while preserving learned weights."""
        runtime.require_symbol("reset_rule_layer")(self._ptr)
