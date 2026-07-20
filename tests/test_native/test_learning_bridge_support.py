# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning bridge test doubles

"""Shared, module-owned native doubles for learning-bridge tests."""

from __future__ import annotations

import ctypes as ct
from typing import Any, Protocol, cast

import pytest

from sc_neurocore._native import learning_runtime as runtime


__test__ = False


class _CValue(Protocol):
    value: int | float


class FakeCFunction:
    """Callable object that accepts ctypes signature metadata."""

    def __init__(self, result: object = None) -> None:
        self.argtypes: list[Any] = []
        self.restype: Any = None
        self.result = result

    def __call__(self, *_args: object) -> object:
        return self.result


class FakeCdll:
    """Dynamic CDLL surface with selectable missing optional symbols."""

    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = set() if missing is None else missing
        self.functions: dict[str, FakeCFunction] = {}

    def __getattr__(self, name: str) -> FakeCFunction:
        if name in self.missing:
            raise AttributeError(name)
        return self.functions.setdefault(name, FakeCFunction())


class FakeLearningLib:
    """State-recording implementation of every wrapper-facing ABI call."""

    def __init__(self) -> None:
        self.rule_ptr = 101
        self.online_ptr = 202
        self.learner_ptr = 303
        self.layer_ptr = 404
        self.wgpu_ptr = 505
        self.state_payload = b"SCAL-state"
        self.state_get_success = True
        self.state_set_success = True
        self.wgpu_set_success = True
        self.destroyed: dict[str, list[int]] = {
            "rule": [],
            "online": [],
            "learner": [],
            "layer": [],
            "wgpu": [],
        }
        self.rule_steps: list[tuple[bool, bool, float, float]] = []
        self.learner_steps: list[tuple[bool, bool, float, float]] = []
        self.batch_counts: list[int] = []
        self.layer_steps: list[float] = []
        self.analog_seeds: list[int] = []
        self.wgpu_seeds: list[int] = []
        self.wgpu_steps: list[float] = []
        self.restored_payloads: list[bytes] = []
        self.restored_weights: list[float] = []

    def create_rule(self, *_args: object) -> int:
        return self.rule_ptr

    def step_rule(
        self, _ptr: int, pre: bool, post: bool, reward: ct.c_float, dt: ct.c_float
    ) -> None:
        self.rule_steps.append((pre, post, float(reward.value), float(dt.value)))

    def step_rule_batched(self, *_args: object) -> None:
        self.batch_counts.append(int(cast(_CValue, _args[-2]).value))

    def get_rule_weight(self, _ptr: int) -> float:
        return 0.75

    def reset_rule(self, _ptr: int) -> None:
        return None

    def destroy_rule(self, ptr: int) -> None:
        self.destroyed["rule"].append(ptr)

    def create_online_o1_synapse(self, *_args: object) -> int:
        return self.online_ptr

    def step_online_o1_synapse(self, *_args: object) -> runtime.OnlineO1SnapshotFFI:
        return runtime.OnlineO1SnapshotFFI(22, 48, 63, 31)

    def online_o1_per_synapse_state_bits(self, _ptr: int) -> int:
        return 26

    def destroy_online_o1_synapse(self, ptr: int) -> None:
        self.destroyed["online"].append(ptr)

    def create_learner(self, *_args: object) -> int:
        return self.learner_ptr

    def step_learner(
        self, _ptr: int, fired: bool, pre: bool, reward: ct.c_float, dt: ct.c_float
    ) -> None:
        self.learner_steps.append((fired, pre, float(reward.value), float(dt.value)))

    def step_learner_batched(self, *_args: object) -> None:
        self.batch_counts.append(int(cast(_CValue, _args[-2]).value))

    def destroy_learner(self, ptr: int) -> None:
        self.destroyed["learner"].append(ptr)

    def create_rule_layer(self, *_args: object) -> int:
        return self.layer_ptr

    def step_rule_layer(self, *_args: object) -> None:
        self.layer_steps.append(float(cast(_CValue, _args[-1]).value))

    def step_rule_layer_analog(self, *_args: object) -> None:
        self.analog_seeds.append(int(cast(_CValue, _args[-2]).value))

    def get_rule_layer_weights(self, _ptr: int, output: Any) -> None:
        for index, value in enumerate((0.1, 0.2, 0.3)):
            output[index] = value

    def destroy_rule_layer(self, ptr: int) -> None:
        self.destroyed["layer"].append(ptr)

    def reset_rule_layer(self, _ptr: int) -> None:
        return None

    def get_rule_layer_state_size(self, _ptr: int) -> int:
        return len(self.state_payload)

    def get_rule_layer_state_mem(self, _ptr: int, output: Any) -> bool:
        for index, value in enumerate(self.state_payload):
            output[index] = value
        return self.state_get_success

    def set_rule_layer_state_mem_checked(self, _ptr: int, source: Any, length: ct.c_size_t) -> bool:
        self.restored_payloads.append(bytes(source[: length.value]))
        return self.state_set_success

    def create_wgpu_layer(self, *_args: object) -> int:
        return self.wgpu_ptr

    def create_wgpu_layer_with_weight(self, *_args: object) -> int:
        return self.wgpu_ptr

    def step_wgpu_layer(self, *_args: object) -> None:
        self.wgpu_steps.append(float(cast(_CValue, _args[-1]).value))

    def get_wgpu_weights(self, _ptr: int, output: Any) -> None:
        for index, value in enumerate((0.4, 0.5, 0.6)):
            output[index] = value

    def set_wgpu_weights(self, _ptr: int, values: Any, count: ct.c_size_t) -> bool:
        self.restored_weights = [float(values[index]) for index in range(count.value)]
        return self.wgpu_set_success

    def set_wgpu_layer_seed(self, _ptr: int, seed: ct.c_uint32) -> None:
        self.wgpu_seeds.append(int(seed.value))

    def reset_wgpu_layer(self, _ptr: int) -> None:
        return None

    def free_wgpu_layer(self, ptr: int) -> None:
        self.destroyed["wgpu"].append(ptr)


@pytest.fixture
def fake_learning_lib(monkeypatch: pytest.MonkeyPatch) -> FakeLearningLib:
    """Install a fresh fake as the authoritative runtime library."""
    fake = FakeLearningLib()
    monkeypatch.setattr(runtime, "_lib", fake)
    monkeypatch.setattr(runtime, "_HAS_LEARNING", True)
    monkeypatch.setattr(runtime, "_DETERMINISTIC_SEED", None)
    return fake
