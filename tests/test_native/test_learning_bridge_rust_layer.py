# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust rule-layer bridge tests

"""Tests for safe Rayon arrays and atomic native state ownership."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore._native import learning_runtime as runtime
from sc_neurocore._native.learning_rust_layer import RustRuleLayer
from sc_neurocore._native.learning_validation import MAX_U64

from test_learning_bridge_support import FakeCdll, FakeLearningLib

pytest_plugins = ("test_learning_bridge_support",)


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "count": 3,
        "rule_type": 1,
        "weight": 0.5,
        "param_a": 0.01,
        "param_b": 0.012,
        "mem_buffer": b"SCAL-state",
    }
    state.update(overrides)
    return state


def test_layer_step_weights_reset_and_close(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustRuleLayer(3)
    layer.step([True, False, True], [False, True, False], [0.1, 0.2, 0.3], dt=0.2)
    assert layer.get_weights() == pytest.approx([0.1, 0.2, 0.3])
    layer.reset()
    layer.close()
    assert fake_learning_lib.layer_steps == [pytest.approx(0.2)]
    assert fake_learning_lib.destroyed["layer"] == [fake_learning_lib.layer_ptr]


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (([True], [False] * 3, [0.0] * 3), "length 3"),
        (([True] * 3, [False], [0.0] * 3), "length 3"),
        (([True] * 3, [False] * 3, [0.0]), "length 3"),
    ],
)
def test_layer_step_rejects_short_buffers(
    fake_learning_lib: FakeLearningLib,
    values: tuple[object, object, object],
    match: str,
) -> None:
    del fake_learning_lib
    layer = RustRuleLayer(3)
    with pytest.raises(ValueError, match=match):
        layer.step(*values)


def test_layer_step_rejects_bad_dt_and_non_finite_rewards(
    fake_learning_lib: FakeLearningLib,
) -> None:
    del fake_learning_lib
    layer = RustRuleLayer(3)
    with pytest.raises(ValueError, match="dt"):
        layer.step([True] * 3, [False] * 3, [0.0] * 3, dt=0.0)
    with pytest.raises(ValueError, match="finite"):
        layer.step([True] * 3, [False] * 3, [0.0, np.nan, 0.0])


def test_layer_analog_seed_precedence_and_wrap(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustRuleLayer(3)
    values = [0.1, 0.5, 0.9]
    layer.step_analog(values, values, [0.0] * 3)
    layer.step_analog(values, values, [0.0] * 3, seed=77)
    runtime.set_deterministic_mode(9)
    layer.step_analog(values, values, [0.0] * 3, seed=88)
    runtime.set_deterministic_mode()
    layer._analog_seed_counter = MAX_U64
    layer.step_analog(values, values, [0.0] * 3)
    assert fake_learning_lib.analog_seeds == [42, 77, 9, MAX_U64]
    assert layer._analog_seed_counter == 0


def test_layer_analog_rejects_invalid_vectors_and_seed(
    fake_learning_lib: FakeLearningLib,
) -> None:
    del fake_learning_lib
    layer = RustRuleLayer(3)
    with pytest.raises(ValueError, match="probabilities"):
        layer.step_analog([0.0, 1.1, 0.0], [0.0] * 3, [0.0] * 3)
    with pytest.raises(ValueError, match="seed"):
        layer.step_analog([0.0] * 3, [0.0] * 3, [0.0] * 3, seed=-1)


def test_state_round_trip_replaces_handle_atomically(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustRuleLayer(3)
    state = layer.get_state_dict()
    assert state["mem_buffer"] == fake_learning_lib.state_payload
    layer.load_state_dict(state)
    assert fake_learning_lib.restored_payloads == [fake_learning_lib.state_payload]
    assert fake_learning_lib.destroyed["layer"] == [fake_learning_lib.layer_ptr]


def test_state_restore_failure_destroys_only_candidate(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustRuleLayer(3)
    original = layer._ptr
    fake_learning_lib.state_set_success = False
    with pytest.raises(ValueError, match="invalid or incompatible"):
        layer.load_state_dict(_state())
    assert layer._ptr == original
    assert fake_learning_lib.destroyed["layer"] == [fake_learning_lib.layer_ptr]


@pytest.mark.parametrize(
    "state",
    [
        {},
        _state(mem_buffer="not bytes"),
        _state(mem_buffer=b""),
        _state(count=0),
        _state(rule_type=8),
        _state(weight=2.0),
        _state(param_a=-1.0),
    ],
)
def test_state_metadata_is_validated_before_native_restore(
    fake_learning_lib: FakeLearningLib, state: dict[str, object]
) -> None:
    del fake_learning_lib
    layer = RustRuleLayer(3)
    with pytest.raises((TypeError, ValueError)):
        layer.load_state_dict(state)


def test_state_requires_length_aware_symbol(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    del fake_learning_lib
    fake = FakeCdll({"set_rule_layer_state_mem_checked"})
    fake.create_rule_layer.result = 123
    monkeypatch.setattr(runtime, "_lib", fake)
    layer = object.__new__(RustRuleLayer)
    with pytest.raises(RuntimeError, match="required symbol"):
        layer.__setstate__(_state())


def test_state_get_failures_are_reported(fake_learning_lib: FakeLearningLib) -> None:
    layer = RustRuleLayer(3)
    fake_learning_lib.state_payload = b""
    with pytest.raises(RuntimeError, match="size query"):
        layer.get_state_dict()
    fake_learning_lib.state_payload = b"SCAL"
    fake_learning_lib.state_get_success = False
    with pytest.raises(RuntimeError, match="serialization"):
        layer.get_state_dict()


def test_unpickle_reload_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    del fake_learning_lib
    monkeypatch.setattr(runtime, "_HAS_LEARNING", False)
    monkeypatch.setattr(runtime, "_load_native_library", lambda: False)
    layer = object.__new__(RustRuleLayer)
    with pytest.raises(RuntimeError, match="not available"):
        layer.__setstate__(_state())


def test_file_state_round_trip_and_io_error(
    fake_learning_lib: FakeLearningLib, tmp_path: Path
) -> None:
    layer = RustRuleLayer(3)
    path = tmp_path / "state.bin"
    assert layer.save(str(path)) is True
    assert path.read_bytes() == fake_learning_lib.state_payload
    assert layer.load(str(path)) is True
    with pytest.raises(FileNotFoundError):
        layer.load(str(tmp_path / "missing.bin"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"rule_type": 9},
        {"weight": -0.1},
        {"param_a": -0.1},
        {"param_b": float("inf")},
    ],
)
def test_layer_constructor_rejects_invalid_domains(
    fake_learning_lib: FakeLearningLib, kwargs: dict[str, object]
) -> None:
    del fake_learning_lib
    with pytest.raises((TypeError, ValueError)):
        RustRuleLayer(**kwargs)  # type: ignore[arg-type]


def test_layer_constructor_rejects_null_handle(fake_learning_lib: FakeLearningLib) -> None:
    fake_learning_lib.layer_ptr = 0
    with pytest.raises(RuntimeError, match="construction failed"):
        RustRuleLayer(3)
