# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native predictive-model backend tests

"""Loader, selection, fail-closed, and FFI marshalling tests for LGSSM backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.world_model import _lgssm_backends as backends
from sc_neurocore.world_model._lgssm_backends import ExplicitBackendName
from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import LinearGaussianSSM


class _DoublePointer(Protocol):
    contents: ctypes.c_double


def _model() -> LinearGaussianSSM:
    return LinearGaussianSSM(
        A=np.eye(2),
        B=np.zeros((2, 0)),
        C=np.eye(2),
        D=np.zeros((2, 0)),
        Q=np.eye(2) * 0.1,
        R=np.eye(2) * 0.2,
        mu_0=np.zeros(2),
        Sigma_0=np.eye(2),
    )


def _inputs() -> tuple[FloatArray, FloatArray]:
    return np.zeros((3, 2)), np.zeros((3, 0))


def _native_mapping() -> dict[str, object]:
    covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    return {
        "means": np.zeros((3, 2)),
        "covariances": covariance,
        "pred_means": np.zeros((3, 2)),
        "pred_covariances": covariance,
        "log_likelihood": -4.0,
    }


def test_missing_rust_filter_raises() -> None:
    with pytest.raises(RuntimeError, match="not available"):
        backends._missing_rust_kalman_filter()


def test_rust_loader_prefers_world_model_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def import_module(name: str) -> object:
        assert name == "sc_neurocore_engine.world_model"
        return SimpleNamespace(get_lgssm_kalman_filter=lambda: sentinel)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert backends._load_rust_kalman_filter() is sentinel


def test_rust_loader_uses_root_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def import_module(name: str) -> object:
        if name == "sc_neurocore_engine.world_model":
            raise ImportError(name)
        return SimpleNamespace(py_lgssm_kalman_filter=sentinel)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert backends._load_rust_kalman_filter() is sentinel


def test_rust_loader_rejects_missing_and_non_callable_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(ImportError, match="not available"):
        backends._load_rust_kalman_filter()

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(get_lgssm_kalman_filter=lambda: object()),
    )
    with pytest.raises(ImportError, match="callable"):
        backends._load_rust_kalman_filter()


def test_module_initialisation_survives_missing_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def import_module_without_rust(
        name: str,
        package: str | None = None,
    ) -> object:
        if name in {"sc_neurocore_engine", "sc_neurocore_engine.world_model"}:
            raise ImportError(name)
        return real_import_module(name, package)

    try:
        with monkeypatch.context() as context:
            context.setattr(importlib, "import_module", import_module_without_rust)
            reloaded = importlib.reload(backends)
            assert reloaded._rust_kalman_filter is None
            assert reloaded._HAS_RUST_LGSSM is False
    finally:
        importlib.reload(backends)


def test_ensure_rust_loaded_caches_success_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def fail_to_load() -> object:
        raise ImportError("missing")

    monkeypatch.setattr(backends, "_rust_kalman_filter", sentinel)
    assert backends._ensure_rust_loaded() is True

    monkeypatch.setattr(backends, "_rust_kalman_filter", None)
    monkeypatch.setattr(
        backends,
        "_load_rust_kalman_filter",
        fail_to_load,
    )
    assert backends._ensure_rust_loaded() is False
    assert backends._HAS_RUST_LGSSM is False

    monkeypatch.setattr(backends, "_load_rust_kalman_filter", lambda: sentinel)
    assert backends._ensure_rust_loaded() is True
    assert backends._HAS_RUST_LGSSM is True


@pytest.mark.parametrize(
    ("backend_name", "relative_path"),
    [
        ("go", Path("accel/go/lgssm/liblgssm.so")),
        ("mojo", Path("accel/mojo/world_model/liblgssm.so")),
    ],
)
def test_c_abi_loader_handles_cache_missing_file_and_load_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    relative_path: Path,
) -> None:
    ensure = cast(Callable[[], bool], getattr(backends, f"_ensure_{backend_name}_loaded"))
    library_attr = f"_{backend_name}_lib"
    flag_attr = f"_HAS_{backend_name.upper()}_LGSSM"
    monkeypatch.setattr(backends, flag_attr, False)
    monkeypatch.setattr(backends, library_attr, object())
    assert ensure() is True
    assert getattr(backends, flag_attr) is True

    monkeypatch.setattr(backends, library_attr, None)
    monkeypatch.setattr(backends, flag_attr, True)
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    assert ensure() is False
    assert getattr(backends, flag_attr) is False

    library_path = tmp_path / relative_path
    library_path.parent.mkdir(parents=True)
    library_path.touch()
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _path: (_ for _ in ()).throw(OSError("bad library")),
    )
    assert ensure() is False


@pytest.mark.parametrize(
    ("backend_name", "relative_path", "argument_count"),
    [
        ("go", Path("accel/go/lgssm/liblgssm.so"), 19),
        ("mojo", Path("accel/mojo/world_model/liblgssm.so"), 19),
    ],
)
def test_c_abi_loader_requires_symbol_and_configures_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    relative_path: Path,
    argument_count: int,
) -> None:
    ensure = cast(Callable[[], bool], getattr(backends, f"_ensure_{backend_name}_loaded"))
    library_attr = f"_{backend_name}_lib"
    flag_attr = f"_HAS_{backend_name.upper()}_LGSSM"
    library_path = tmp_path / relative_path
    library_path.parent.mkdir(parents=True)
    library_path.touch()
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backends, library_attr, None)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    assert ensure() is False

    class FakeFunction:
        argtypes: list[object] | None = None
        restype: object | None = object()

        def __call__(self, *_args: object) -> None:
            return None

    function = FakeFunction()
    library = SimpleNamespace(kalman_filter_c=function)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: library)
    assert ensure() is True
    assert len(function.argtypes or []) == argument_count
    assert function.restype is None
    assert getattr(backends, library_attr) is library
    assert getattr(backends, flag_attr) is True


def test_julia_loader_handles_cache_dependency_file_and_module_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(backends, "_julia_module", object())
    monkeypatch.setattr(backends, "_HAS_JULIA_LGSSM", False)
    assert backends._ensure_julia_loaded() is True
    assert backends._HAS_JULIA_LGSSM is True

    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(backends, "_HAS_JULIA_LGSSM", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert backends._ensure_julia_loaded() is False
    assert backends._HAS_JULIA_LGSSM is False

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    assert backends._ensure_julia_loaded() is False

    module_path = tmp_path / "accel/julia/world_model/predictive_model.jl"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(Main=SimpleNamespace()),
    )
    assert backends._ensure_julia_loaded() is False

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("Julia init failed")),
    )
    assert backends._ensure_julia_loaded() is False


def test_julia_loader_includes_module_and_caches_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_module = object()

    class Main:
        PredictiveModelAccel = loaded_module
        included: str | None = None

        @classmethod
        def include(cls, path: str) -> None:
            cls.included = path

    module_path = tmp_path / "accel/julia/world_model/predictive_model.jl"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(Main=Main),
    )

    assert backends._ensure_julia_loaded() is True
    assert backends._julia_module is loaded_module
    assert Main.included == str(module_path)
    assert backends._HAS_JULIA_LGSSM is True


def test_probe_and_resolve_backend_follow_fastest_first_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability = {"mojo": False, "rust": False, "julia": True, "go": True}
    monkeypatch.setattr(backends, "_ensure_mojo_loaded", lambda: availability["mojo"])
    monkeypatch.setattr(backends, "_ensure_rust_loaded", lambda: availability["rust"])
    monkeypatch.setattr(backends, "_ensure_julia_loaded", lambda: availability["julia"])
    monkeypatch.setattr(backends, "_ensure_go_loaded", lambda: availability["go"])

    assert backends.probe_backend("python") == (True, "")
    assert backends.probe_backend("mojo")[0] is False
    assert "Mojo" in backends.probe_backend("mojo")[1]
    assert backends.resolve_backend("auto") == "go"
    assert backends.resolve_backend("go") == "go"

    availability["go"] = False
    with pytest.raises(RuntimeError, match="Go LGSSM backend"):
        backends.resolve_backend("go")
    with pytest.raises(ValueError, match="backend must be"):
        backends.resolve_backend("cuda")


def test_auto_resolution_reaches_python_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backends,
        "probe_backend",
        lambda backend: (backend == "python", "missing"),
    )
    assert backends.resolve_backend("auto") == "python"


def test_auto_resolution_fails_closed_if_even_python_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "probe_backend",
        lambda _backend: (False, "missing"),
    )
    with pytest.raises(RuntimeError, match="no executable LGSSM backend"):
        backends.resolve_backend("auto")


def test_rust_marshalling_returns_validated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def rust_filter(**kwargs: object) -> object:
        captured.update(kwargs)
        return _native_mapping()

    monkeypatch.setattr(backends, "_rust_kalman_filter", rust_filter)
    observations, controls = _inputs()
    result = backends.filter_native("rust", _model(), observations, controls)

    assert captured["t_len"] == 3
    assert captured["p_dim"] == 2
    assert captured["m_dim"] == 0
    assert len(cast(list[float], captured["a_flat"])) == 4
    assert result.means.shape == (3, 2)
    assert result.log_likelihood == -4.0


def test_rust_marshalling_rejects_non_numeric_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _native_mapping()
    mapping["log_likelihood"] = object()
    monkeypatch.setattr(backends, "_rust_kalman_filter", lambda **_kwargs: mapping)
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="non-numeric log_likelihood"):
        backends.filter_native("rust", _model(), observations, controls)


def test_rust_marshalling_rejects_out_of_range_numeric_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _native_mapping()
    mapping["log_likelihood"] = 10**10_000
    monkeypatch.setattr(backends, "_rust_kalman_filter", lambda **_kwargs: mapping)
    observations, controls = _inputs()

    with pytest.raises(RuntimeError, match="out-of-range log_likelihood"):
        backends.filter_native("rust", _model(), observations, controls)


@pytest.mark.parametrize(
    "malformed_result",
    [
        object(),
        {"means": np.zeros((3, 2))},
        {
            **_native_mapping(),
            "means": object(),
        },
        {
            **_native_mapping(),
            "means": [[10**10_000]],
        },
        {
            **_native_mapping(),
            "means": np.zeros((2, 2)),
        },
        {
            "means": np.zeros((2, 2)),
            "covariances": np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            "pred_means": np.zeros((2, 2)),
            "pred_covariances": np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            "log_likelihood": -1.0,
        },
    ],
)
def test_rust_marshalling_rejects_malformed_payloads(
    monkeypatch: pytest.MonkeyPatch,
    malformed_result: object,
) -> None:
    monkeypatch.setattr(
        backends,
        "_rust_kalman_filter",
        lambda **_kwargs: malformed_result,
    )
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="returned|missing"):
        backends.filter_native("rust", _model(), observations, controls)


def test_julia_marshalling_reads_named_result_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = _native_mapping()
    result_object = SimpleNamespace(
        means=mapping["means"],
        covariances=mapping["covariances"],
        pred_means=mapping["pred_means"],
        pred_covs=mapping["pred_covariances"],
        log_lik=mapping["log_likelihood"],
    )

    class JuliaModule:
        @staticmethod
        def kalman_filter(*_args: object) -> object:
            return result_object

    monkeypatch.setattr(backends, "_julia_module", JuliaModule())
    observations, controls = _inputs()
    result = backends.filter_native("julia", _model(), observations, controls)
    assert result.log_likelihood == -4.0


def test_julia_marshalling_rejects_incomplete_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JuliaModule:
        @staticmethod
        def kalman_filter(*_args: object) -> object:
            return SimpleNamespace(means=np.zeros((3, 2)))

    monkeypatch.setattr(backends, "_julia_module", JuliaModule())
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="incomplete filter result"):
        backends.filter_native("julia", _model(), observations, controls)


def _write_c_outputs(
    time_steps: int,
    state_dim: int,
    means_address: int,
    covariances_address: int,
    pred_means_address: int,
    pred_covariances_address: int,
    likelihood_address: int,
) -> None:
    means = ctypes.cast(means_address, ctypes.POINTER(ctypes.c_double))
    covariances = ctypes.cast(covariances_address, ctypes.POINTER(ctypes.c_double))
    pred_means = ctypes.cast(pred_means_address, ctypes.POINTER(ctypes.c_double))
    pred_covariances = ctypes.cast(
        pred_covariances_address,
        ctypes.POINTER(ctypes.c_double),
    )
    likelihood = ctypes.cast(likelihood_address, ctypes.POINTER(ctypes.c_double))
    for index in range(time_steps * state_dim):
        means[index] = 0.0
        pred_means[index] = 0.0
    for time_index in range(time_steps):
        for row in range(state_dim):
            for column in range(state_dim):
                index = time_index * state_dim * state_dim + row * state_dim + column
                value = 1.0 if row == column else 0.0
                covariances[index] = value
                pred_covariances[index] = value
    likelihood[0] = -4.0


def test_mojo_raw_address_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    class MojoLibrary:
        @staticmethod
        def kalman_filter_c(*args: object) -> None:
            values = [cast(int, value) for value in args]
            _write_c_outputs(values[10], values[13], *values[14:19])

    monkeypatch.setattr(backends, "_mojo_lib", MojoLibrary())
    observations, controls = _inputs()
    result = backends.filter_native("mojo", _model(), observations, controls)
    assert result.covariances.shape == (3, 2, 2)
    assert result.log_likelihood == -4.0


def test_go_pointer_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    class GoLibrary:
        @staticmethod
        def kalman_filter_c(*args: object) -> None:
            time_steps = cast(ctypes.c_int, args[10]).value
            state_dim = cast(ctypes.c_int, args[13]).value
            addresses = [
                ctypes.addressof(
                    cast(_DoublePointer, pointer).contents,
                )
                for pointer in args[14:19]
            ]
            _write_c_outputs(time_steps, state_dim, *addresses)

    monkeypatch.setattr(backends, "_go_lib", GoLibrary())
    observations, controls = _inputs()
    result = backends.filter_native("go", _model(), observations, controls)
    assert result.pred_covariances.shape == (3, 2, 2)
    assert result.log_likelihood == -4.0


@pytest.mark.parametrize(
    ("backend", "attribute", "message"),
    [
        ("rust", "_rust_kalman_filter", "Rust backend was selected"),
        ("julia", "_julia_module", "Julia backend was selected"),
        ("go", "_go_lib", "Go backend was selected"),
        ("mojo", "_mojo_lib", "Mojo backend was selected"),
    ],
)
def test_native_runners_fail_closed_without_loaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
    backend: ExplicitBackendName,
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setattr(backends, attribute, None)
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match=message):
        backends.filter_native(backend, _model(), observations, controls)


def test_filter_native_rejects_python_backend() -> None:
    observations, controls = _inputs()
    with pytest.raises(ValueError, match="cannot execute the Python backend"):
        backends.filter_native("python", _model(), observations, controls)
