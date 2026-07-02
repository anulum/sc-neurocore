# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage for autonomous learning ctypes bridge branches

from __future__ import annotations

import builtins
import ctypes as ct
import importlib.util
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore._native import learning_bridge as lb


class _FakeCFunction:
    def __init__(self) -> None:
        self.argtypes: list[Any] = []
        self.restype: Any = None

    def __call__(self, *_args: object) -> object:
        return None


class _FakeCdllWithoutOnlineO1:
    _ONLINE_SYMBOLS = {
        "create_online_o1_synapse",
        "step_online_o1_synapse",
        "online_o1_per_synapse_state_bits",
        "destroy_online_o1_synapse",
    }

    def __init__(self) -> None:
        self.functions: dict[str, _FakeCFunction] = {}

    def __getattr__(self, name: str) -> _FakeCFunction:
        if name in self._ONLINE_SYMBOLS:
            raise AttributeError(name)
        function = self.functions.setdefault(name, _FakeCFunction())
        return function


class _FakeLearningLib:
    def __init__(self) -> None:
        self.destroyed_rules: list[int] = []
        self.destroyed_online_o1: list[int] = []
        self.destroyed_learners: list[int] = []
        self.destroyed_layers: list[int] = []
        self.destroyed_wgpu_layers: list[int] = []
        self.saved_buffers: list[bytes] = []
        self.rule_steps: list[tuple[bool, bool, float, float]] = []
        self.learner_steps: list[tuple[bool, bool, float, float]] = []
        self.layer_steps: list[float] = []
        self.reset_layers: list[int] = []
        self.analog_seeds: list[int] = []
        self.wgpu_seeds: list[int] = []
        self.save_success = True
        self.load_success = True
        self.wgpu_ptr = 999

    def create_rule(self, *_args: object) -> int:
        return 101

    def step_rule(
        self,
        _ptr: int,
        pre_spike: bool,
        post_spike: bool,
        reward: ct.c_float,
        dt: ct.c_float,
    ) -> None:
        self.rule_steps.append((pre_spike, post_spike, float(reward.value), float(dt.value)))
        return None

    def get_rule_weight(self, _ptr: int) -> float:
        return 0.75

    def reset_rule(self, *_args: object) -> None:
        return None

    def destroy_rule(self, ptr: int) -> None:
        self.destroyed_rules.append(int(ptr))

    def create_online_o1_synapse(self, *_args: object) -> int:
        return 404

    def step_online_o1_synapse(self, *_args: object) -> lb.OnlineO1SnapshotFFI:
        return lb.OnlineO1SnapshotFFI(weight=22, pre_trace=48, post_trace=63, eligibility=31)

    def online_o1_per_synapse_state_bits(self, _ptr: int) -> int:
        return 26

    def destroy_online_o1_synapse(self, ptr: int) -> None:
        self.destroyed_online_o1.append(int(ptr))

    def create_learner(self, *_args: object) -> int:
        return 202

    def step_learner(
        self,
        _ptr: int,
        fired: bool,
        pre_spike: bool,
        global_reward: ct.c_float,
        dt: ct.c_float,
    ) -> None:
        self.learner_steps.append((fired, pre_spike, float(global_reward.value), float(dt.value)))
        return None

    def destroy_learner(self, ptr: int) -> None:
        self.destroyed_learners.append(int(ptr))

    def step_rule_batched(self, *_args: object) -> None:
        return None

    def step_learner_batched(self, *_args: object) -> None:
        return None

    def create_rule_layer(self, *_args: object) -> int:
        return 303

    def step_rule_layer(
        self,
        _ptr: int,
        _pre_ptr: Any,
        _post_ptr: Any,
        _rew_ptr: Any,
        dt: ct.c_float,
    ) -> None:
        self.layer_steps.append(float(dt.value))
        return None

    def get_rule_layer_weights(self, _ptr: int, out_ptr: Any) -> None:
        for i, value in enumerate((0.1, 0.2, 0.3)):
            out_ptr[i] = value

    def destroy_rule_layer(self, ptr: int) -> None:
        self.destroyed_layers.append(int(ptr))

    def reset_rule_layer(self, ptr: int) -> None:
        self.reset_layers.append(int(ptr))
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
        _pre_ptr: Any,
        _post_ptr: Any,
        _rew_ptr: Any,
        seed: ct.c_uint64,
        _dt: ct.c_float,
    ) -> None:
        self.analog_seeds.append(int(seed.value))

    def create_wgpu_layer(self, *_args: object) -> int:
        return self.wgpu_ptr

    def step_wgpu_layer(self, *_args: object) -> None:
        return None

    def get_wgpu_weights(self, _ptr: int, out_ptr: Any) -> None:
        for i, value in enumerate((0.4, 0.5, 0.6)):
            out_ptr[i] = value

    def set_wgpu_layer_seed(self, _ptr: int, seed: ct.c_uint32) -> None:
        self.wgpu_seeds.append(int(seed.value))

    def free_wgpu_layer(self, ptr: int) -> None:
        self.destroyed_wgpu_layers.append(int(ptr))

    def reset_wgpu_layer(self, *_args: object) -> None:
        return None


class _FakeLearningLibNoOnlineO1:
    def create_rule(self, *_args: object) -> int:
        return 101


@pytest.fixture(autouse=True)
def _restore_learning_bridge_state() -> Generator[None, None, None]:
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


def _set_native_lib(fake: object | None) -> None:
    lb._lib = cast(Any, fake)


def test_get_lib_raises_when_unloaded() -> None:
    lb._lib = None
    with pytest.raises(RuntimeError, match="not loaded"):
        lb._get_lib()


def test_load_native_library_returns_false_for_missing_env_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(object())
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
    monkeypatch.setattr(ct, "CDLL", lambda _path: (_ for _ in ()).throw(OSError("boom")))
    assert lb._load_native_library() is False
    assert lb.is_available() is False


def test_load_native_library_accepts_legacy_library_without_online_o1_symbols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_lib = tmp_path / "libautonomous_learning.so"
    fake_lib.write_bytes(b"")
    fake_cdll = _FakeCdllWithoutOnlineO1()
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(fake_lib))
    monkeypatch.setattr(ct, "CDLL", lambda _path: fake_cdll)

    assert lb._load_native_library() is True
    assert lb.is_available() is True
    assert cast(object, lb._lib) is fake_cdll
    assert "create_rule" in fake_cdll.functions
    assert "create_online_o1_synapse" not in fake_cdll.functions


def test_rule_constructors_raise_when_native_unavailable() -> None:
    lb._HAS_LEARNING = False
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustOnlineO1Synapse()
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustPlasticityRule()
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustEligentLearner()
    with pytest.raises(RuntimeError, match="not available"):
        lb.RustRuleLayer(count=3)
    with pytest.raises(RuntimeError, match="not loaded"):
        lb.RustWgpuRuleLayer(count=3)


def test_online_o1_constructor_rejects_missing_symbols_and_null_handles() -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLibNoOnlineO1())
    with pytest.raises(RuntimeError, match="lacks online O\\(1\\) symbols"):
        lb.RustOnlineO1Synapse()

    fake = _FakeLearningLib()
    fake.create_online_o1_synapse = lambda *_args: 0  # type: ignore[method-assign]
    _set_native_lib(fake)
    with pytest.raises(ValueError, match="invalid online O\\(1\\)"):
        lb.RustOnlineO1Synapse()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_bits": 0},
        {"weight_bits": 32},
        {"trace_bits": 1},
        {"trace_bits": 31},
        {"reward_bits": 0},
        {"reward_bits": 31},
        {"learning_shift": 31},
        {"trace_decay_shift": 31},
        {"initial_weight": -1},
    ],
)
def test_online_o1_bridge_rejects_out_of_range_domains_before_ctypes(
    kwargs: dict[str, Any],
) -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLib())

    with pytest.raises(ValueError):
        lb.RustOnlineO1Synapse(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_bits": True},
        {"trace_bits": False},
        {"reward_bits": 4.0},
        {"learning_shift": 2.0},
        {"trace_decay_shift": "2"},
        {"initial_weight": 1.5},
    ],
)
def test_online_o1_bridge_rejects_bool_and_non_integral_domains_before_ctypes(
    kwargs: dict[str, Any],
) -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLib())

    with pytest.raises(TypeError):
        lb.RustOnlineO1Synapse(**kwargs)


def test_rule_and_learner_batched_length_guards() -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLib())

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


def test_rule_and_learner_valid_batched_paths_and_reset() -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLib())

    rule = lb.RustPlasticityRule()
    learner = lb.RustEligentLearner()

    rule.step_batched(
        np.array([True, False]),
        np.array([False, True]),
        np.array([0.1, -0.2], dtype=np.float32),
        dt=0.002,
    )
    learner.step_batched(
        np.array([True, False]),
        np.array([False, True]),
        np.array([0.1, -0.2], dtype=np.float32),
        dt=0.002,
    )
    rule.reset()


def test_rule_and_learner_single_step_forward_dt_to_native() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

    rule = lb.RustPlasticityRule()
    learner = lb.RustEligentLearner()

    rule.step(pre_spike=True, post_spike=False, dt=0.003, reward=-0.25)
    learner.step(fired=False, pre_spike=True, global_reward=0.75, dt=0.004)

    assert fake.rule_steps[0][:2] == (True, False)
    assert fake.rule_steps[0][2] == pytest.approx(-0.25)
    assert fake.rule_steps[0][3] == pytest.approx(0.003)
    assert fake.learner_steps[0][:2] == (False, True)
    assert fake.learner_steps[0][2] == pytest.approx(0.75)
    assert fake.learner_steps[0][3] == pytest.approx(0.004)


def test_rule_destructors_release_handles() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

    rule = lb.RustPlasticityRule()
    learner = lb.RustEligentLearner()
    assert rule.weight == pytest.approx(0.75)

    rule.__del__()
    learner.__del__()

    assert fake.destroyed_rules == [101]
    assert fake.destroyed_learners == [202]


def test_online_o1_bridge_wraps_bounded_rust_kernel() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

    synapse = lb.RustOnlineO1Synapse(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
        initial_weight=0,
    )

    snapshot = synapse.step(pre_spike=False, post_spike=True, reward=-7)

    assert snapshot.weight == 22
    assert snapshot.pre_trace == 48
    assert snapshot.post_trace == 63
    assert snapshot.eligibility == 31
    assert synapse.per_synapse_state_bits == 26

    synapse.__del__()
    assert fake.destroyed_online_o1 == [404]


@pytest.mark.parametrize("reward", [True, 1.5, "2"])
def test_online_o1_bridge_rejects_non_integral_reward_before_ctypes(reward: Any) -> None:
    lb._HAS_LEARNING = True
    _set_native_lib(_FakeLearningLib())

    synapse = lb.RustOnlineO1Synapse()
    try:
        with pytest.raises(TypeError, match="reward"):
            synapse.step(pre_spike=True, post_spike=False, reward=reward)
    finally:
        synapse.__del__()


def test_rule_layer_state_roundtrip_and_file_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

    layer = lb.RustRuleLayer(count=3)
    state = layer.get_state_dict()
    assert state["mem_buffer"] == b"ABCD"

    restored = object.__new__(lb.RustRuleLayer)
    restored.__setstate__(state)
    assert fake.saved_buffers[-1] == b"ABCD"

    restored.load_state_dict(state)
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
    restored.__del__()
    assert fake.destroyed_layers[-1] == 303

    monkeypatch.setattr(lb, "_load_native_library", lambda: False)
    lb._HAS_LEARNING = False
    _set_native_lib(None)
    restored_missing = object.__new__(lb.RustRuleLayer)
    with pytest.raises(RuntimeError, match="not available"):
        restored_missing.__setstate__(state)


def test_rule_layer_step_and_reset_forward_to_native() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

    layer = lb.RustRuleLayer(count=3)
    spikes = np.array([True, False, True])
    rewards = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    layer.step(spikes, spikes, rewards, dt=0.007)
    layer.reset()

    assert fake.layer_steps == [pytest.approx(0.007)]
    assert fake.reset_layers == [303]


def test_rule_layer_step_analog_seed_paths() -> None:
    fake = _FakeLearningLib()
    lb._HAS_LEARNING = True
    _set_native_lib(fake)

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
    _set_native_lib(fake)
    lb.set_deterministic_mode(77)

    layer = lb.RustWgpuRuleLayer(count=3)
    assert fake.wgpu_seeds == [77]

    spikes = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    layer.step(spikes, spikes, rewards=None)
    layer.step_analog(spikes, spikes, spikes)
    assert np.allclose(layer.get_weights(), np.array([0.4, 0.5, 0.6], dtype=np.float32))
    assert np.allclose(
        layer.get_state_dict()["weights"],
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )

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


def test_torch_non_autograd_step_state_and_reward_warning() -> None:
    torch = pytest.importorskip("torch")

    layer = lb.create_plasticity_layer(
        count=3,
        rule_type=lb.RULE_REWARD_STDP,
        backend="torch",
        autograd=False,
    )
    pre = np.array([True, False, True])
    post = np.array([False, True, True])
    rewards = np.array([0.2, -0.1, 0.3], dtype=np.float32)

    with pytest.warns(UserWarning, match="expects 'rewards'"):
        _ = layer.forward(torch.tensor(pre), torch.tensor(post), rewards=None, dt=1.0)
    layer.step(pre, post, rewards, dt=1.0)

    state = layer.get_state_dict()
    clone = lb.create_plasticity_layer(
        count=3,
        rule_type=lb.RULE_REWARD_STDP,
        backend="torch",
        autograd=False,
    )
    clone.load_state_dict(state)

    assert np.all(np.isfinite(layer.get_weights()))
    assert np.allclose(clone.get_weights(), layer.get_weights())


def test_torch_precision_accepts_tensor_and_numpy_bit_specs() -> None:
    torch = pytest.importorskip("torch")

    layer = lb.create_plasticity_layer(
        count=4,
        rule_type=lb.RULE_REWARD_STDP,
        backend="torch",
        autograd=False,
        weight=0.41,
        weight_bits=torch.tensor([2, 3, 4, 5]),
        trace_bits=np.array([3, 4, 5, 6]),
        eligibility_bits=4,
        weight_clip=1.0,
        trace_clip=1.0,
        eligibility_clip=1.0,
    )

    pre = torch.tensor([1.0, 0.0, 1.0, 0.0])
    post = torch.tensor([0.0, 1.0, 0.0, 1.0])
    rewards = torch.ones(4)
    layer.forward(pre, post, rewards, dt=1.0)

    assert torch.all(torch.isfinite(layer.weights))
    assert torch.all(torch.isfinite(layer.pre_trace))
    assert torch.all(torch.isfinite(layer.post_trace))
    assert torch.all(torch.isfinite(layer.eligibility))


def test_torch_precision_rejects_sub_two_bits_and_non_positive_clips() -> None:
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="weight_bits entries"):
        lb.create_plasticity_layer(
            count=4,
            rule_type=lb.RULE_STDP,
            backend="torch",
            autograd=False,
            weight_bits=[1, 2, 3, 4],
        )

    with pytest.raises(ValueError, match="weight_bits must be scalar or have length 4"):
        lb.create_plasticity_layer(
            count=4,
            rule_type=lb.RULE_STDP,
            backend="torch",
            autograd=False,
            weight_bits=[2, 3, 4],
        )

    with pytest.raises(ValueError, match="weight_clip"):
        lb.create_plasticity_layer(
            count=4,
            rule_type=lb.RULE_STDP,
            backend="torch",
            autograd=False,
            weight_clip=0.0,
        )


@pytest.mark.parametrize(
    "rule_type",
    [lb.RULE_STDP, lb.RULE_REWARD_STDP, lb.RULE_BCM, lb.RULE_ELIGENT],
)
def test_torch_rule_layer_reset_matches_native_rule_scope(rule_type: int) -> None:
    torch = pytest.importorskip("torch")

    layer = lb.create_plasticity_layer(
        count=3,
        rule_type=rule_type,
        backend="torch",
        autograd=False,
    )

    with torch.no_grad():
        layer.pre_trace.fill_(0.2)
        layer.post_trace.fill_(0.3)
        layer.eligibility.fill_(0.4)
        layer.theta_m.fill_(0.9)
        layer.act_avg.fill_(0.8)

    layer.reset()

    if rule_type == lb.RULE_STDP:
        assert torch.count_nonzero(layer.pre_trace).item() == 0
        assert torch.count_nonzero(layer.post_trace).item() == 0
        assert torch.all(layer.eligibility == 0.4)
    elif rule_type == lb.RULE_REWARD_STDP:
        assert torch.count_nonzero(layer.pre_trace).item() == 0
        assert torch.count_nonzero(layer.post_trace).item() == 0
        assert torch.count_nonzero(layer.eligibility).item() == 0
    elif rule_type == lb.RULE_BCM:
        assert torch.count_nonzero(layer.act_avg).item() == 0
        assert torch.all(layer.theta_m == 0.5)
        assert torch.all(layer.pre_trace == 0.2)
    else:
        assert torch.count_nonzero(layer.eligibility).item() == 0
        assert torch.all(layer.pre_trace == 0.2)
        assert torch.all(layer.post_trace == 0.3)


@pytest.mark.parametrize(
    "rule_type",
    [lb.RULE_STDP, lb.RULE_REWARD_STDP, lb.RULE_ELIGENT, lb.RULE_BCM],
)
def test_torch_autograd_backward_routes_rule_specific_gradients(rule_type: int) -> None:
    torch = pytest.importorskip("torch")

    layer = lb.create_plasticity_layer(
        count=3,
        rule_type=rule_type,
        backend="torch",
        autograd=True,
    )
    pre = torch.tensor([1.0, 0.0, 1.0], requires_grad=True)
    post = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
    rewards = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)

    loss = layer.forward(pre, post, rewards, dt=1.0).sum()
    loss.backward()

    assert layer.weights.grad is not None
    if rule_type in (lb.RULE_STDP, lb.RULE_REWARD_STDP, lb.RULE_ELIGENT, lb.RULE_BCM):
        assert pre.grad is not None
    if rule_type in (lb.RULE_STDP, lb.RULE_REWARD_STDP, lb.RULE_BCM):
        assert post.grad is not None
    if rule_type in (lb.RULE_REWARD_STDP, lb.RULE_ELIGENT):
        assert rewards.grad is not None


def test_learning_bridge_torch_unavailable_factory_reports_install_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path(lb.__file__)
    spec = importlib.util.spec_from_file_location("_learning_bridge_no_torch", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def reject_torch(
        name: str,
        global_vars: Mapping[str, object] | None = None,
        local_vars: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for fallback test")
        return real_import(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    spec.loader.exec_module(module)

    with pytest.raises(ImportError, match="requires PyTorch"):
        module.create_plasticity_layer(count=1)
