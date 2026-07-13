# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning runtime tests

"""Tests for atomic loading, ABI typing, and deterministic runtime state."""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any
from collections.abc import Generator

import pytest

from sc_neurocore._native import learning_runtime as runtime

from test_learning_bridge_support import FakeCdll


@pytest.fixture(autouse=True)
def _restore_runtime_state() -> Generator[None, None, None]:
    """Keep loader-state tests isolated from real native integration tests."""
    library = runtime._lib
    available = runtime._HAS_LEARNING
    seed = runtime._DETERMINISTIC_SEED
    try:
        yield
    finally:
        runtime._lib = library
        runtime._HAS_LEARNING = available
        runtime._DETERMINISTIC_SEED = seed


def test_library_path_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", "/tmp/explicit-learning.so")
    assert runtime._library_path() == Path("/tmp/explicit-learning.so")


@pytest.mark.parametrize(
    ("system", "name"),
    [
        ("Windows", "autonomous_learning.dll"),
        ("Darwin", "libautonomous_learning.dylib"),
        ("Linux", "libautonomous_learning.so"),
    ],
)
def test_library_path_selects_platform(
    monkeypatch: pytest.MonkeyPatch, system: str, name: str
) -> None:
    monkeypatch.delenv("SC_NEUROCORE_LIB_PATH", raising=False)
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    assert runtime._library_path().name == name


def test_loader_missing_path_clears_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(tmp_path / "missing.so"))
    monkeypatch.setattr(runtime, "_lib", object())
    monkeypatch.setattr(runtime, "_HAS_LEARNING", True)
    assert runtime._load_native_library() is False
    assert runtime._lib is None
    assert runtime.is_available() is False


def test_loader_types_complete_abi_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "learning.so"
    path.write_bytes(b"test-double")
    fake = FakeCdll()
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(path))
    monkeypatch.setattr(ct, "CDLL", lambda _path: fake)
    assert runtime._load_native_library() is True
    assert runtime._lib is fake
    assert runtime.is_available() is True
    assert fake.create_rule.restype is ct.c_void_p
    assert fake.step_rule_layer_analog.argtypes[-2:] == [ct.c_uint64, ct.c_float]
    assert fake.set_rule_layer_state_mem_checked.restype is ct.c_bool
    assert fake.create_wgpu_layer_with_weight.restype is ct.c_void_p
    assert fake.step_online_o1_synapse.restype is runtime.OnlineO1SnapshotFFI


@pytest.mark.parametrize(
    "missing",
    [
        {"set_rule_layer_state_mem_checked"},
        {"create_wgpu_layer_with_weight"},
        {"create_online_o1_synapse"},
    ],
)
def test_loader_accepts_absent_optional_extensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: set[str]
) -> None:
    path = tmp_path / "legacy.so"
    path.touch()
    fake = FakeCdll(missing)
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(path))
    monkeypatch.setattr(ct, "CDLL", lambda _path: fake)
    assert runtime._load_native_library() is True
    assert runtime.is_available() is True


def test_loader_rejects_partial_required_abi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "partial.so"
    path.touch()
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(path))
    monkeypatch.setattr(ct, "CDLL", lambda _path: FakeCdll({"create_rule"}))
    assert runtime._load_native_library() is False
    assert runtime._lib is None
    assert runtime._HAS_LEARNING is False


def test_loader_handles_dynamic_loader_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "broken.so"
    path.touch()
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", str(path))

    def fail(_path: str) -> Any:
        raise OSError("bad ELF")

    monkeypatch.setattr(ct, "CDLL", fail)
    assert runtime._load_native_library() is False
    assert runtime._lib is None


def test_runtime_queries_and_required_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_lib", None)
    monkeypatch.setattr(runtime, "_HAS_LEARNING", True)
    assert runtime.is_available() is False
    assert runtime.has_symbol("anything") is False
    with pytest.raises(RuntimeError, match="not loaded"):
        runtime._get_lib()

    fake = FakeCdll({"absent"})
    monkeypatch.setattr(runtime, "_lib", fake)
    assert runtime.has_symbol("present") is True
    assert runtime.require_symbol("present") is fake.present
    with pytest.raises(RuntimeError, match="required symbol absent"):
        runtime.require_symbol("absent")


class _Destroyer:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[object] = []

    def destroy(self, pointer: object) -> None:
        self.calls.append(pointer)
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("error", [AttributeError(), OSError(), TypeError()])
def test_destroy_noexcept_swallows_ffi_teardown_errors(
    monkeypatch: pytest.MonKeyPatch, error: BaseException
) -> None:
    fake = _Destroyer(error)
    monkeypatch.setattr(runtime, "_lib", fake)
    runtime.destroy_noexcept("destroy", 7)
    assert fake.calls == [7]


def test_destroy_noexcept_handles_absence_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_lib", None)
    runtime.destroy_noexcept("destroy", 7)
    fake = _Destroyer()
    monkeypatch.setattr(runtime, "_lib", fake)
    runtime.destroy_noexcept("destroy", None)
    runtime.destroy_noexcept("destroy", 8)
    assert fake.calls == [8]


def test_deterministic_seed_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_DETERMINISTIC_SEED", None)
    runtime.set_deterministic_mode(42)
    assert runtime.deterministic_seed() == 42
    runtime.set_deterministic_mode()
    assert runtime.deterministic_seed() is None
    with pytest.raises((TypeError, ValueError)):
        runtime.set_deterministic_mode(True)
