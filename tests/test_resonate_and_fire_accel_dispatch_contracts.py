# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire accelerator dispatch contracts

"""Cover selection, validation, reload, and malformed-result boundaries."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import resonate_and_fire as backends
from sc_neurocore.accel.go import resonate_and_fire as go_backend
from sc_neurocore.accel.mojo import resonate_and_fire as mojo_backend
from tests.module_reload import preserve_module_identity

_PARAMETERS = (0.13, -0.27, -0.8, 7.5, 0.9, 0.006)


def _baseline(steps: int = 2) -> dict[str, Any]:
    drive = np.linspace(3.0, 4.0, steps, dtype=np.float64)
    return dict(backends.simulate_python(*_PARAMETERS, drive))


def _spiking_baseline() -> dict[str, Any]:
    return dict(backends.simulate_python(0.0, 0.99, 0.0, 1.0, 1.0, 0.1, [10.0]))


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_resonate_and_fire(
            *_PARAMETERS,
            np.zeros((2, 2)),
            backend="python",
        )
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_resonate_and_fire(
            *_PARAMETERS,
            [1.5, np.nan],
            backend="python",
        )


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
    with pytest.raises(ValueError, match="unknown resonate-and-fire backend"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, [1.5], backend="fortran")
    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Rust resonate-and-fire backend is unavailable"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, [1.5], backend="rust")


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
    class JuliaError(Exception):
        """Stand in for JuliaCall's optional runtime exception type."""

    monkeypatch.setattr(
        backends,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(JuliaError("Julia startup failed")),
    )
    assert not backends.backend_available("julia")

    monkeypatch.setattr(
        backends,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(RuntimeError("unrelated Python defect")),
    )
    with pytest.raises(RuntimeError, match="unrelated Python defect"):
        backends.backend_available("julia")

    def missing_native(_backend: str) -> Any:
        raise ImportError("native module absent")

    monkeypatch.setattr(backends, "_native_module", missing_native)
    assert not backends.backend_available("go")


def test_missing_engine_export_disables_rust_without_breaking_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_simulate_python = backends.simulate_python
    real_import_module = importlib.import_module

    def import_without_export(name: str) -> Any:
        if name == "sc_neurocore_engine":
            return SimpleNamespace()
        return real_import_module(name)

    with preserve_module_identity(backends), monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", import_without_export)
        reloaded = importlib.reload(backends)
        assert not reloaded.backend_available("rust")
        assert reloaded.backend_available("python")

    assert backends.simulate_python is original_simulate_python
    assert np.asarray(backends.simulate_python(*_PARAMETERS, [1.5])["x"]).shape == (1,)


@pytest.mark.parametrize("key", ("x", "y", "spikes"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result = _baseline()
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        backends.normalise_result(
            result,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )


def test_result_validator_rejects_nonfinite_and_nonbinary_traces() -> None:
    nonfinite = _baseline()
    nonfinite["y"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite y"):
        backends.normalise_result(
            nonfinite,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )

    nonbinary = _baseline()
    nonbinary["spikes"] = np.asarray([0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-binary"):
        backends.normalise_result(
            nonbinary,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )


def test_result_validator_rejects_missing_trace_and_final_receipt() -> None:
    missing_trace = _baseline()
    del missing_trace["x"]
    with pytest.raises(FloatingPointError, match="invalid x trace"):
        backends.normalise_result(
            missing_trace,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )

    missing_final = _baseline()
    del missing_final["y_final"]
    with pytest.raises(FloatingPointError, match="invalid y_final"):
        backends.normalise_result(
            missing_final,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )


def test_result_validator_enforces_final_trace_consistency() -> None:
    result = _baseline()
    result["x_final"] = float(result["x_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="x_final disagrees"):
        backends.normalise_result(
            result,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )


@pytest.mark.parametrize("bad_count", (True, 1.0, np.nan, "0", -1))
def test_result_validator_requires_consistent_integral_spike_count(
    bad_count: object,
) -> None:
    result = _baseline()
    result["spike_count"] = bad_count
    with pytest.raises(FloatingPointError, match="spike_count"):
        backends.normalise_result(
            result,
            n_steps=2,
            initial=_PARAMETERS[:2],
            threshold=_PARAMETERS[4],
        )


def test_result_validator_rejects_reset_receipt_drift() -> None:
    wrong_x = _spiking_baseline()
    wrong_x["x"] = np.asarray([0.25])
    wrong_x["x_final"] = 0.25
    with pytest.raises(FloatingPointError, match="x reset"):
        backends.normalise_result(
            wrong_x,
            n_steps=1,
            initial=(0.0, 0.99),
            threshold=1.0,
        )

    wrong_y = _spiking_baseline()
    wrong_y["y"] = np.asarray([0.75])
    wrong_y["y_final"] = 0.75
    with pytest.raises(FloatingPointError, match="y reset"):
        backends.normalise_result(
            wrong_y,
            n_steps=1,
            initial=(0.0, 0.99),
            threshold=1.0,
        )


def test_native_runner_rechecks_rust_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust resonate-and-fire backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.zeros((2, 2)))
    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libresonate_and_fire.so not built"):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.asarray([1.5]))


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
        module.simulate_resonate_and_fire(*_PARAMETERS, oversized)


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_library_probe_handles_loader_failure(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_path: str) -> None:
        raise OSError("shared library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_to_load)
    assert module._load_library() == (None, False)


@pytest.mark.parametrize(
    ("status", "exception", "message"),
    (
        (1, RuntimeError, "code 1"),
        (2, ValueError, "configuration"),
        (3, ValueError, "current"),
        (4, FloatingPointError, "candidate"),
    ),
)
@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_maps_each_native_status(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    exception: type[Exception],
    message: str,
) -> None:
    fake = SimpleNamespace(resonate_and_fire_simulate_c=lambda *_args: status)
    monkeypatch.setattr(module, "_lib", fake)
    with pytest.raises(exception, match=message):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.asarray([1.5]))
