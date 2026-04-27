# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage for autonomous learning ctypes bridge branches

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore._native import learning_bridge as lb


class _FakeLearningLib:
    def __init__(self) -> None:
        self.destroyed_rules: list[int] = []
        self.destroyed_learners: list[int] = []
        self.destroyed_layers: list[int] = []
        self.destroyed_wgpu_layers: list[int] = []
        self.saved_buffers: list[bytes] = []
        self.analog_seeds: list[int] = []
        self.wgpu_seeds: list[int] = []
        self.save_success = True
        self.load_success = True
        self.wgpu_ptr = 999

    def create_rule(self, *_args: object) -> int:
        return 101

    def step_rule(self, *_args: object) -> None:
        return None

    def get_rule_weight(self, _ptr: int) -> float:
        return 0.75

    def reset_rule(self, *_args: object) -> None:
        return None

    def destroy_rule(self, ptr: int) -> None:
        self.destroyed_rules.append(int(ptr))

    def create_learner(self, *_args: object) -> int:
        return 202

    def step_learner(self, *_args: object) -> None:
        return None

    def destroy_learner(self, ptr: int) -> None:
        self.destroyed_learners.append(int(ptr))

    def step_rule_batched(self, *_args: object) -> None:
        return None

    def step_learner_batched(self, *_args: object) -> None:
        return None

    def create_rule_layer(self, *_args: object) -> int:
        return 303

    def step_rule_layer(self, *_args: object) -> None:
        return None

    def get_rule_layer_weights(self, _ptr: int, out_ptr: ct.POINTER(ct.c_float)) -> None:
        for i, value in enumerate((0.1, 0.2, 0.3)):
            out_ptr[i] = value

    def destroy_rule_layer(self, ptr: int) -> None:
        self.destroyed_layers.append(int(ptr))

    def reset_rule_layer(self, *_args: object) -> None:
        return None

    def save_rule_layer_batched(self, *_args: object) -> bool:
        return self.save_success

    def load_rule_layer_batched(self, *_args: object) -> bool:
        return self.load_success

    def get_rule_layer_state_size(self, _ptr: int) -> int:
        return 4

    def get_rule_layer_state_mem(self, _ptr: int, buf: ct.Array[ct.c_byte]) -> bool:
        for i, value in enumerate(b"ABCD"):
            buf[i] = value
        return True

    def set_rule_layer_state_mem(self, _ptr: int, buf: ct.Array[ct.c_byte]) -> bool:
        self.saved_buffers.append(bytes(buf))
        return True

    def step_rule_layer_analog(
        self,
        _ptr: int,
        _pre_ptr: ct.POINTER(ct.c_float),
        _post_ptr: ct.POINTER(ct.c_float),
        _rew_ptr: ct.POINTER(ct.c_float),
        seed: ct.c_uint64,
        _dt: ct.c_float,
    ) -> None:
        self.analog_seeds.append(int(seed.value))

    def create_wgpu_layer(self, *_args: object) -> int:
        return self.wgpu_ptr

    def step_wgpu_layer(self, *_args: object) -> None:
        return None

    def get_wgpu_weights(self, _ptr: int, out_ptr: ct.POINTER(ct.c_float)) -> None:
        for i, value in enumerate((0.4, 0.5, 0.6)):
            out_ptr[i] = value

    def set_wgpu_layer_seed(self, _ptr: int, seed: ct.c_uint32) -> None:
        self.wgpu_seeds.append(int(seed.value))

    def free_wgpu_layer(self, ptr: int) -> None:
        self.destroyed_wgpu_layers.append(int(ptr))

    def reset_wgpu_layer(self, *_args: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _restore_learning_bridge_state() -> None:
    old_has = lb._HAS_LEARNING
    old_lib = lb._lib
    old_seed = lb._DETERMINISTIC_SEED
    old_env = Path.cwd()
    try:
        yield
    finally:
        lb._HAS_LEARNING = old_has
        lb._lib = old_lib
        lb._DETERMINISTIC_SEED = old_seed
        del old_env


def test_get_lib_raises_when_unloaded() -> None:
    lb._lib = None
    with pytest.raises(RuntimeError, match="not loaded"):
        lb._get_lib()


def test_load_native_library_returns_false_for_missing_env_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lb._HAS_LEARNING = True
    lb._lib = object()
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", "/definitely/missing/libautonomous_learning.so")
    assert lb._load_native_library() is False
    assert lb.is_available() is False
    assert lb._lib is None


def test_load_native_library_returns_false_on_cdll_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_lib = tmp_path / "libautonomous_learning.so"
    fake_lib.write_bytes(b"not-a-real-library")
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(fake_lib))
    monkeypatch.setattr(lb._ct, "CDLL", lambda _path: (_ for _ in ()).throw(OSError("boom")))
    assert lb._load_native_library() is False
    assert lb.is_available() is False


def test_rule_constructors_raise_when_native_unavailable() -> None:
    lb._HAS_LEARNING = False
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustPlasticityRule()
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustEligentLearner()
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustRuleLayer(count=3)
    with pytest.raises(RuntimeError, match="not loaded"):
        lb.RustWgpuRuleLayer(count=3)


def test_rule_and_learner_batched_length_guards() -> None:
    lb._HAS_LEARNING = True
    lb._lib = _FakeLearningLib()

    rule = lb.RustPlasticityRule()
    with pytest.raises(ValueError, match="mismatch"):
        rule.step_batched(
            np.array([True, False]),
            np.array([True]),
            np.array([0.1, 0.2], dtype=np.float32),
        )

    learner = lb.RustEligentLearner()
    with pytest.raises(ValueError, match="identically sized"):
        learner.step_batched(
            np.array([True, False]),
            np.array([True]),
            np.array([0.1, 0.2], dtype=np.float32),
        )


def test_rule_destructors_release_handles() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    lb._lib = fake

    rule = lb.RustPlasticityRule()
    learner = lb.RustEligentLearner()
    assert rule.weight == pytest.approx(0.75)

    rule.__del__()
    learner.__del__()

    assert fake.destroyed_rules == [101]
    assert fake.destroyed_learners == [202]


def test_rule_layer_state_roundtrip_and_file_paths(tmp_path: Path) -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    lb._lib = fake

    layer = lb.RustRuleLayer(count=3)
    state = layer.get_state_dict()
    assert state["mem_buffer"] == b"ABCD"

    restored = object.__new__(lb.RustRuleLayer)
    restored.__setstate__(state)
    assert fake.saved_buffers[-1] == b"ABCD"

    weights = layer.get_weights()
    assert np.allclose(weights, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    assert layer.save(str(tmp_path / "traits.bin")) is True
    assert layer.load(str(tmp_path / "traits.bin")) is True

    fake.save_success = False
    with pytest.raises(OSError, match="Failed to save"):
        layer.save(str(tmp_path / "traits.bin"))

    fake.load_success = False
    with pytest.raises(OSError, match="Failed to load"):
        layer.load(str(tmp_path / "traits.bin"))

    layer.__del__()
    assert fake.destroyed_layers[-1] == 303


def test_rule_layer_step_analog_seed_paths() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    lb._lib = fake

    layer = lb.RustRuleLayer(count=3)
    probs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    rewards = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    layer.step_analog(probs, probs, rewards)
    layer.step_analog(probs, probs, rewards)
    assert fake.analog_seeds[:2] == [42, 43]

    lb.set_deterministic_mode(1234)
    layer.step_analog(probs, probs, rewards)
    assert fake.analog_seeds[-1] == 1234


def test_wgpu_layer_paths_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    lb._lib = fake
    lb.set_deterministic_mode(77)

    layer = lb.RustWgpuRuleLayer(count=3)
    assert fake.wgpu_seeds == [77]

    spikes = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    layer.step(spikes, spikes, rewards=None)
    layer.step_analog(spikes, spikes, spikes)
    assert np.allclose(layer.get_weights(), np.array([0.4, 0.5, 0.6], dtype=np.float32))

    with pytest.warns(UserWarning, match="load_state_dict"):
        layer.load_state_dict({"weights": np.array([1.0], dtype=np.float32)})

    layer.reset()
    layer.__del__()
    assert fake.destroyed_wgpu_layers == [999]

    rust_layer = lb.create_plasticity_layer(count=2, backend="rust")
    assert isinstance(rust_layer, lb.RustRuleLayer)

    wgpu_layer = lb.create_plasticity_layer(count=2, backend="rust-wgpu")
    assert isinstance(wgpu_layer, lb.RustWgpuRuleLayer)

    fake.wgpu_ptr = 0
    with pytest.raises(RuntimeError, match="failed"):
        lb.RustWgpuRuleLayer(count=2)

    with pytest.raises(ValueError, match="Unknown backend"):
        lb.create_plasticity_layer(count=2, backend="bogus")
