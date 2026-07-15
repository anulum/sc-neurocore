# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen–Rit accelerator dispatch contracts

"""Cover selection, validation, and malformed-native-result boundaries."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from sc_neurocore.accel import jansen_rit as backends
from sc_neurocore.accel.go import jansen_rit as go_backend
from sc_neurocore.accel.mojo import jansen_rit as mojo_backend

_PARAMETERS = (
    0.1,
    0.2,
    0.3,
    -0.4,
    -0.1,
    0.5,
    3.25,
    22.0,
    100.0,
    50.0,
    135.0,
    2.5,
    6.0,
    0.56,
    0.0001,
)


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_jansen_rit(*_PARAMETERS, np.zeros((2, 2)), backend="python")
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_jansen_rit(*_PARAMETERS, [220.0, np.nan], backend="python")


def test_unknown_and_explicitly_unavailable_backends_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown Jansen–Rit backend"):
        backends.simulate_jansen_rit(*_PARAMETERS, [220.0], backend="fortran")
    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Rust Jansen–Rit backend is unavailable"):
        backends.simulate_jansen_rit(*_PARAMETERS, [220.0], backend="rust")


def test_auto_selection_uses_first_available_measured_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("mojo", "go", "rust", "julia", static[-1]),
    )
    monkeypatch.setattr(backends, "backend_available", lambda backend: backend in {"go", "python"})
    assert backends.auto_backend() == "go"


def test_python_floor_is_always_available() -> None:
    assert backends.backend_available("python")
    assert not backends.backend_available("unknown")


def test_optional_backend_discovery_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable_julia() -> None:
        raise RuntimeError("Julia startup failed")

    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda _name: SimpleNamespace(_ensure_jansen_rit_loaded=unavailable_julia),
    )
    assert not backends.backend_available("julia")

    def missing_native(_backend: str) -> Any:
        raise ImportError("native module absent")

    monkeypatch.setattr(backends, "_native_module", missing_native)
    assert not backends.backend_available("go")


@pytest.mark.parametrize("key", ("y0", "eeg"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:6])


def test_result_validator_rejects_nonfinite_trace() -> None:
    result: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    result["y4"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite y4"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:6])


def test_result_validator_rejects_missing_trace() -> None:
    result: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    del result["y0"]
    with pytest.raises(FloatingPointError, match="invalid y0 trace"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:6])


def test_result_validator_enforces_eeg_identity() -> None:
    result: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    result["eeg"] = np.asarray(result["eeg"]) + 1.0
    with pytest.raises(FloatingPointError, match="EEG trace disagrees"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:6])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    result["y2_final"] = float(result["y2_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="y2_final disagrees"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:6])


def test_result_validator_rejects_missing_or_nonfinite_final_state() -> None:
    missing: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    del missing["y0_final"]
    with pytest.raises(FloatingPointError, match="invalid y0_final"):
        backends.normalise_result(missing, n_steps=2, initial=_PARAMETERS[:6])

    nonfinite: dict[str, Any] = dict(
        backends.simulate_python(*_PARAMETERS, np.asarray([120.0, 220.0]))
    )
    nonfinite["y3_final"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite y3_final"):
        backends.normalise_result(nonfinite, n_steps=2, initial=_PARAMETERS[:6])


def test_native_runner_rechecks_rust_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust Jansen–Rit backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_jansen_rit(*_PARAMETERS, np.zeros((2, 2)))

    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libjansen_rit.so not built"):
        module.simulate_jansen_rit(*_PARAMETERS, np.asarray([220.0]))


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_library_probe_handles_loader_failure(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_path: str) -> None:
        raise OSError("shared library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_to_load)
    assert module._load_library() == (None, False)
