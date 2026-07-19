# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse accelerator dispatch contracts

"""Cover selection, validation, reload, and malformed-result boundaries."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.accel import alpha as backends
from sc_neurocore.accel.go import alpha as go_backend
from sc_neurocore.accel.mojo import alpha as mojo_backend
from sc_neurocore.neurons.models.alpha import AlphaResult
from tests.module_reload import preserve_module_identity

_PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)


def _baseline(steps: int = 2) -> dict[str, object]:
    exc = np.linspace(2.0, 2.5, steps, dtype=np.float64)
    inh = np.linspace(0.5, 0.75, steps, dtype=np.float64)
    return dict(backends.simulate_python(*_PARAMETERS, exc, inh))


def _normalise(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float, float, float, float],
) -> AlphaResult:
    return backends.normalise_result(result, n_steps=n_steps, initial=initial, v_rest=-0.5)


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_alpha(*_PARAMETERS, np.zeros((2, 2)), backend="python")
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_alpha(*_PARAMETERS, [1.5, np.nan], backend="python")
    with pytest.raises(ValueError, match="scalar or match"):
        backends.simulate_alpha(*_PARAMETERS, [1.5, 2.0], [0.1, 0.2, 0.3], backend="python")


def test_signed_32_bit_step_bound_precedes_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        backends._input(oversized, 0.0)


def test_unknown_and_explicitly_unavailable_backends_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown alpha backend"):
        backends.simulate_alpha(*_PARAMETERS, [1.5], backend="fortran")
    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Rust alpha backend is unavailable"):
        backends.simulate_alpha(*_PARAMETERS, [1.5], backend="rust")


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
        lambda: (_ for _ in ()).throw(ImportError("Julia runtime absent")),
    )
    assert not backends.backend_available("julia")

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
    assert np.asarray(backends.simulate_python(*_PARAMETERS, [1.5])["v"]).shape == (1,)


@pytest.mark.parametrize("key", ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result = _baseline()
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:5])


def test_result_validator_rejects_nonfinite_and_nonbinary_traces() -> None:
    nonfinite = _baseline()
    nonfinite["i_exc"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite i_exc"):
        _normalise(nonfinite, n_steps=2, initial=_PARAMETERS[:5])

    nonbinary = _baseline()
    nonbinary["spikes"] = np.asarray([0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-binary"):
        _normalise(nonbinary, n_steps=2, initial=_PARAMETERS[:5])


def test_result_validator_rejects_missing_trace_and_final_receipt() -> None:
    missing_trace = _baseline()
    del missing_trace["a_exc"]
    with pytest.raises(FloatingPointError, match="invalid a_exc trace"):
        _normalise(missing_trace, n_steps=2, initial=_PARAMETERS[:5])

    missing_final = _baseline()
    del missing_final["i_inh_final"]
    with pytest.raises(FloatingPointError, match="invalid i_inh_final"):
        _normalise(missing_final, n_steps=2, initial=_PARAMETERS[:5])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result = _baseline()
    result["v_final"] = cast(float, result["v_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="v_final disagrees"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:5])


def test_result_validator_rejects_nonfinite_final_and_missing_spike_count() -> None:
    nonfinite_final = _baseline()
    nonfinite_final["a_inh_final"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite a_inh_final"):
        _normalise(nonfinite_final, n_steps=2, initial=_PARAMETERS[:5])

    missing_count = _baseline()
    del missing_count["spike_count"]
    with pytest.raises(FloatingPointError, match="invalid spike_count"):
        _normalise(missing_count, n_steps=2, initial=_PARAMETERS[:5])


@pytest.mark.parametrize("bad_count", (True, 1.0, np.nan, "0", -1))
def test_result_validator_requires_consistent_integral_spike_count(
    bad_count: object,
) -> None:
    result = _baseline()
    result["spike_count"] = bad_count
    with pytest.raises(FloatingPointError, match="spike_count"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:5])


def test_result_validator_rejects_reset_receipt_drift() -> None:
    wrong = _baseline()
    wrong["spikes"] = np.asarray([1.0, 0.0])
    wrong["v"] = np.asarray([-0.3, -0.3])
    wrong["v_final"] = -0.3
    wrong["spike_count"] = 1
    with pytest.raises(FloatingPointError, match="somatic v reset"):
        _normalise(wrong, n_steps=2, initial=_PARAMETERS[:5])


def test_native_runner_rechecks_rust_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust alpha backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_alpha(*_PARAMETERS, np.zeros((2, 2)))
    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libalpha.so not built"):
        module.simulate_alpha(*_PARAMETERS, np.asarray([1.5]))


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
        module.simulate_alpha(*_PARAMETERS, oversized)


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
    fake = SimpleNamespace(alpha_simulate_c=lambda *_args: status)
    monkeypatch.setattr(module, "_lib", fake)
    with pytest.raises(exception, match=message):
        module.simulate_alpha(*_PARAMETERS, np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75]))
