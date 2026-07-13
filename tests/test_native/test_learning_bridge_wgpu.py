# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WGPU learning-wrapper tests

"""Tests for WGPU construction, exact host buffers, seeds, and state."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_runtime as runtime
from sc_neurocore._native.learning_wgpu import RustWgpuRuleLayer

from test_learning_bridge_support import FakeCdll, FakeLearningLib

pytest_plugins = ("test_learning_bridge_support",)


def test_wgpu_full_lifecycle_and_state(fake_learning_lib: FakeLearningLib) -> None:
    runtime.set_deterministic_mode(19)
    layer = RustWgpuRuleLayer(3, weight=0.25)
    layer.step([True, False, True], [False, True, False], dt=0.2)
    layer.step_analog([0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.0] * 3, seed=88)
    assert layer.get_weights() == pytest.approx([0.4, 0.5, 0.6])
    assert layer.get_state_dict()["weights"] == pytest.approx([0.4, 0.5, 0.6])
    layer.load_state_dict({"weights": [0.2, 0.3, 0.4]})
    layer.reset()
    layer.close()
    assert fake_learning_lib.wgpu_seeds == [19, 19]
    assert fake_learning_lib.wgpu_steps == [pytest.approx(0.2), pytest.approx(0.001)]
    assert fake_learning_lib.restored_weights == pytest.approx([0.2, 0.3, 0.4])
    assert fake_learning_lib.destroyed["wgpu"] == [fake_learning_lib.wgpu_ptr]


def test_wgpu_analog_uses_explicit_seed_without_global(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustWgpuRuleLayer(3)
    layer.step_analog([0.0] * 3, [0.0] * 3, [0.0] * 3, seed=7)
    layer.step_analog([0.0] * 3, [0.0] * 3, [0.0] * 3)
    assert fake_learning_lib.wgpu_seeds == [7]


def test_wgpu_legacy_constructor_accepts_only_historical_weight(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    monkeypatch.delattr(FakeLearningLib, "create_wgpu_layer_with_weight")
    layer = RustWgpuRuleLayer(3)
    assert layer._ptr == fake_learning_lib.wgpu_ptr
    with pytest.raises(RuntimeError, match="cannot set a WGPU initial weight"):
        RustWgpuRuleLayer(3, weight=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"rule_type": 9},
        {"weight": 1.1},
        {"param_a": -0.1},
        {"param_b": float("nan")},
        {"tau_e": 0.0},
        {"target_sum_weights": -1.0},
        {"tau_plus": 0.0},
        {"tau_minus": float("inf")},
    ],
)
def test_wgpu_constructor_rejects_invalid_domains(
    fake_learning_lib: FakeLearningLib, kwargs: dict[str, object]
) -> None:
    del fake_learning_lib
    with pytest.raises((TypeError, ValueError)):
        RustWgpuRuleLayer(**kwargs)  # type: ignore[arg-type]


def test_wgpu_constructor_rejects_missing_library_and_null_handle(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    fake_learning_lib.wgpu_ptr = 0
    with pytest.raises(RuntimeError, match="initialization failed"):
        RustWgpuRuleLayer(3)
    monkeypatch.setattr(runtime, "_HAS_LEARNING", False)
    monkeypatch.setattr(runtime, "_lib", None)
    with pytest.raises(RuntimeError, match="not available"):
        RustWgpuRuleLayer(3)


@pytest.mark.parametrize(
    ("pre", "post", "reward", "match"),
    [
        ([True], [False] * 3, None, "length 3"),
        ([True] * 3, [False], None, "length 3"),
        ([True] * 3, [False] * 3, [0.0], "length 3"),
        ([0.0, 1.1, 0.0], [False] * 3, None, "probabilities"),
        ([True] * 3, [False] * 3, [0.0, np.nan, 0.0], "finite"),
    ],
)
def test_wgpu_step_rejects_unsafe_host_buffers(
    fake_learning_lib: FakeLearningLib,
    pre: object,
    post: object,
    reward: object,
    match: str,
) -> None:
    del fake_learning_lib
    layer = RustWgpuRuleLayer(3)
    with pytest.raises(ValueError, match=match):
        layer.step(pre, post, reward)


def test_wgpu_step_rejects_bad_dt_and_seed(fake_learning_lib: FakeLearningLib) -> None:
    del fake_learning_lib
    layer = RustWgpuRuleLayer(3)
    with pytest.raises(ValueError, match="dt"):
        layer.step([False] * 3, [False] * 3, dt=0.0)
    with pytest.raises(ValueError, match="seed"):
        layer.step_analog([0.0] * 3, [0.0] * 3, [0.0] * 3, seed=-1)


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"weights": [0.1]},
        {"weights": [-0.1, 0.2, 0.3]},
        {"weights": [0.1, 0.2, 1.1]},
    ],
)
def test_wgpu_state_rejects_invalid_payloads(
    fake_learning_lib: FakeLearningLib, state: dict[str, object]
) -> None:
    del fake_learning_lib
    layer = RustWgpuRuleLayer(3)
    with pytest.raises((ValueError, TypeError)):
        layer.load_state_dict(state)


def test_wgpu_state_reports_missing_symbol_and_native_failure(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    layer = RustWgpuRuleLayer(3)
    fake_learning_lib.wgpu_set_success = False
    with pytest.raises(RuntimeError, match="restoration failed"):
        layer.load_state_dict({"weights": [0.1, 0.2, 0.3]})

    fake = FakeCdll({"set_wgpu_weights"})
    fake.create_wgpu_layer_with_weight.result = 12
    monkeypatch.setattr(runtime, "_lib", fake)
    replacement = RustWgpuRuleLayer(3)
    with pytest.raises(RuntimeError, match="required symbol"):
        replacement.load_state_dict({"weights": [0.1, 0.2, 0.3]})
