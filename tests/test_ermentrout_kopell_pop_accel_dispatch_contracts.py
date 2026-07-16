# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR accelerator dispatch contracts

"""Cover selection, validation, and malformed-native-result boundaries."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import ermentrout_kopell_pop as backends
from sc_neurocore.accel.go import ermentrout_kopell_pop as go_backend
from sc_neurocore.accel.mojo import ermentrout_kopell_pop as mojo_backend
from tests.module_reload import preserve_module_identity

_PARAMETERS = (0.13, -1.7, 1.3, 0.8, -4.2, 12.5, 0.004)


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_ermentrout_kopell_pop(*_PARAMETERS, np.zeros((2, 2)), backend="python")
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_ermentrout_kopell_pop(*_PARAMETERS, [1.5, np.nan], backend="python")


def test_signed_32_bit_step_bound_precedes_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        backends._input(oversized)


def test_unknown_and_explicitly_unavailable_backends_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown MPR backend"):
        backends.simulate_ermentrout_kopell_pop(*_PARAMETERS, [1.5], backend="fortran")
    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Rust MPR backend is unavailable"):
        backends.simulate_ermentrout_kopell_pop(*_PARAMETERS, [1.5], backend="rust")


def test_auto_selection_uses_first_available_measured_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("mojo", "go", "rust", "julia", static[-1]),
    )
    monkeypatch.setattr(
        backends,
        "backend_available",
        lambda backend: backend in {"go", "python"},
    )
    assert backends.auto_backend() == "go"


def test_python_floor_is_always_available() -> None:
    assert backends.backend_available("python")
    assert not backends.backend_available("unknown")


def test_optional_backend_discovery_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ControlledJuliaError(Exception):
        pass

    def unavailable_julia() -> None:
        raise ControlledJuliaError("Julia startup failed")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            _ensure_ermentrout_kopell_pop_loaded=unavailable_julia,
            is_julia_error=lambda error: isinstance(error, ControlledJuliaError),
        ),
    )
    assert not backends.backend_available("julia")

    def unrelated_failure() -> None:
        raise RuntimeError("unrelated Python defect")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            _ensure_ermentrout_kopell_pop_loaded=unrelated_failure,
            is_julia_error=lambda _error: False,
        ),
    )
    with pytest.raises(RuntimeError, match="unrelated Python defect"):
        backends.backend_available("julia")

    def missing_native(_backend: str) -> Any:
        raise ImportError("native module absent")

    monkeypatch.setattr(backends, "_native_module", missing_native)
    assert not backends.backend_available("go")


def test_missing_engine_export_disables_rust_without_breaking_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_simulate_python = backends.simulate_python
    real_import_module = importlib.import_module

    def import_without_mpr_export(name: str) -> Any:
        if name == "sc_neurocore_engine":
            return SimpleNamespace()
        return real_import_module(name)

    with preserve_module_identity(backends), monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", import_without_mpr_export)
        reloaded = importlib.reload(backends)
        assert not reloaded.backend_available("rust")
        assert reloaded.backend_available("python")

    assert backends.simulate_python is original_simulate_python
    restored = backends.simulate_python(*_PARAMETERS, np.asarray([1.5]))
    assert np.asarray(restored["r"]).shape == (1,)


@pytest.mark.parametrize("key", ("r", "v"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_nonfinite_or_negative_rate_trace() -> None:
    nonfinite: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    nonfinite["v"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite v"):
        backends.normalise_result(nonfinite, n_steps=2, initial=_PARAMETERS[:2])

    negative: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    negative["r"] = np.asarray([0.1, -0.1])
    with pytest.raises(FloatingPointError, match="negative firing-rate"):
        backends.normalise_result(negative, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_missing_trace() -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    del result["r"]
    with pytest.raises(FloatingPointError, match="invalid r trace"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    result["v_final"] = float(result["v_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="v_final disagrees"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_missing_or_nonfinite_final_state() -> None:
    missing: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    del missing["r_final"]
    with pytest.raises(FloatingPointError, match="invalid r_final"):
        backends.normalise_result(missing, n_steps=2, initial=_PARAMETERS[:2])

    nonfinite: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    nonfinite["v_final"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite v_final"):
        backends.normalise_result(nonfinite, n_steps=2, initial=_PARAMETERS[:2])

    negative: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    negative["r_final"] = -0.1
    with pytest.raises(FloatingPointError, match="negative final firing rate"):
        backends.normalise_result(negative, n_steps=2, initial=_PARAMETERS[:2])


def test_native_runner_rechecks_rust_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust MPR backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, np.zeros((2, 2)))

    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libermentrout_kopell_pop.so not built"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, np.asarray([1.5]))


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_step_bound_precedes_contiguous_copy(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(module.np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, oversized)


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_library_probe_handles_loader_failure(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_path: str) -> None:
        raise OSError("shared library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_to_load)
    assert module._load_library() == (None, False)
