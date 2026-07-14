# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native LGSSM backend loading and marshalling

"""Load and execute the maintained native forward Kalman-filter backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Literal, Protocol, cast

import numpy as np

from ._lgssm_types import FilterResult, FloatArray, LinearGaussianSSM

BackendName = Literal["auto", "mojo", "go", "rust", "julia", "python"]
ExplicitBackendName = Literal["mojo", "go", "rust", "julia", "python"]

AUTO_BACKEND_ORDER: tuple[ExplicitBackendName, ...] = (
    "mojo",
    "go",
    "rust",
    "julia",
    "python",
)

_RustKalmanFilter = Callable[..., object]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _CFunction(Protocol):
    argtypes: list[object] | None
    restype: object | None

    def __call__(self, *args: object) -> object: ...


class _CLibrary(Protocol):
    kalman_filter_c: _CFunction


class _JuliaResult(Protocol):
    means: object
    covariances: object
    pred_means: object
    pred_covs: object
    log_lik: object


class _JuliaModule(Protocol):
    def kalman_filter(
        self,
        observations: FloatArray,
        controls: FloatArray,
        A: FloatArray,
        B: FloatArray,
        C: FloatArray,
        D: FloatArray,
        Q: FloatArray,
        R: FloatArray,
        mu_0: FloatArray,
        Sigma_0: FloatArray,
    ) -> _JuliaResult: ...


def _missing_rust_kalman_filter(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("Rust LGSSM backend is not available")


def _load_rust_kalman_filter() -> _RustKalmanFilter:
    try:
        world_model_module = importlib.import_module("sc_neurocore_engine.world_model")
        candidate = world_model_module.get_lgssm_kalman_filter()
    except (AttributeError, ImportError):
        try:
            engine_module = importlib.import_module("sc_neurocore_engine")
            candidate = engine_module.py_lgssm_kalman_filter
        except (AttributeError, ImportError) as exc:
            raise ImportError("Rust LGSSM backend is not available") from exc
    if not callable(candidate):
        raise ImportError("Rust LGSSM backend did not expose a callable filter")
    return cast(_RustKalmanFilter, candidate)


try:
    _rust_kalman_filter: _RustKalmanFilter | None = _load_rust_kalman_filter()
    _HAS_RUST_LGSSM = True
except ImportError:
    _rust_kalman_filter = None
    _HAS_RUST_LGSSM = False

_julia_module: _JuliaModule | None = None
_HAS_JULIA_LGSSM = False
_go_lib: _CLibrary | None = None
_HAS_GO_LGSSM = False
_mojo_lib: _CLibrary | None = None
_HAS_MOJO_LGSSM = False


def _ensure_rust_loaded() -> bool:
    global _rust_kalman_filter, _HAS_RUST_LGSSM
    if _rust_kalman_filter is not None:
        _HAS_RUST_LGSSM = True
        return True
    try:
        _rust_kalman_filter = _load_rust_kalman_filter()
    except ImportError:
        _HAS_RUST_LGSSM = False
        return False
    _HAS_RUST_LGSSM = True
    return True


def _ensure_mojo_loaded() -> bool:
    """Load the Mojo C-ABI library once and validate its exported symbol."""
    global _mojo_lib, _HAS_MOJO_LGSSM
    if _mojo_lib is not None:
        _HAS_MOJO_LGSSM = True
        return True
    _HAS_MOJO_LGSSM = False
    library_path = _PACKAGE_ROOT / "accel" / "mojo" / "world_model" / "liblgssm.so"
    if not library_path.is_file():
        return False
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return False
    function = getattr(library, "kalman_filter_c", None)
    if function is None:
        return False
    typed_function = cast(_CFunction, function)
    typed_function.argtypes = [ctypes.c_int64] * 19
    typed_function.restype = None
    _mojo_lib = cast(_CLibrary, library)
    _HAS_MOJO_LGSSM = True
    return True


def _ensure_go_loaded() -> bool:
    """Load the Go C-ABI library once and validate its exported symbol."""
    global _go_lib, _HAS_GO_LGSSM
    if _go_lib is not None:
        _HAS_GO_LGSSM = True
        return True
    _HAS_GO_LGSSM = False
    library_path = _PACKAGE_ROOT / "accel" / "go" / "lgssm" / "liblgssm.so"
    if not library_path.is_file():
        return False
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return False
    function = getattr(library, "kalman_filter_c", None)
    if function is None:
        return False
    typed_function = cast(_CFunction, function)
    typed_function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    typed_function.restype = None
    _go_lib = cast(_CLibrary, library)
    _HAS_GO_LGSSM = True
    return True


def _ensure_julia_loaded() -> bool:
    """Load the Julia module once without paying startup cost for unused paths."""
    global _julia_module, _HAS_JULIA_LGSSM
    if _julia_module is not None:
        _HAS_JULIA_LGSSM = True
        return True
    _HAS_JULIA_LGSSM = False
    if importlib.util.find_spec("juliacall") is None:
        return False
    module_path = _PACKAGE_ROOT / "accel" / "julia" / "world_model" / "predictive_model.jl"
    if not module_path.is_file():
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        main = juliacall.Main
        main.include(str(module_path))
        loaded_module = main.PredictiveModelAccel
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False
    _julia_module = cast(_JuliaModule, loaded_module)
    _HAS_JULIA_LGSSM = True
    return True


_UNAVAILABLE_REASONS: dict[ExplicitBackendName, str] = {
    "python": "",
    "rust": (
        "Rust LGSSM backend requested but the sc_neurocore_engine filter is unavailable; "
        "install a matching sc-neurocore-engine wheel."
    ),
    "julia": (
        "Julia LGSSM backend requested but juliacall or the predictive-model module is unavailable."
    ),
    "go": (
        "Go LGSSM backend requested but liblgssm.so is unavailable; build "
        "src/sc_neurocore/accel/go/lgssm/lgssm.go as a C shared library."
    ),
    "mojo": (
        "Mojo LGSSM backend requested but liblgssm.so is unavailable; build "
        "src/sc_neurocore/accel/mojo/world_model/lgssm.mojo as a shared library."
    ),
}


def probe_backend(backend: ExplicitBackendName) -> tuple[bool, str]:
    """Return runtime availability and a precise unavailability reason."""
    loaders: dict[ExplicitBackendName, Callable[[], bool]] = {
        "python": lambda: True,
        "rust": _ensure_rust_loaded,
        "julia": _ensure_julia_loaded,
        "go": _ensure_go_loaded,
        "mojo": _ensure_mojo_loaded,
    }
    available = loaders[backend]()
    return available, "" if available else _UNAVAILABLE_REASONS[backend]


def resolve_backend(backend: str) -> ExplicitBackendName:
    """Resolve an explicit or fastest-first automatic backend selection."""
    valid_names = ("auto", *AUTO_BACKEND_ORDER)
    if backend not in valid_names:
        choices = "/".join(valid_names)
        raise ValueError(f"backend must be {choices}, got {backend!r}")
    if backend == "auto":
        for candidate in AUTO_BACKEND_ORDER:
            if probe_backend(candidate)[0]:
                return candidate
        raise RuntimeError("no executable LGSSM backend is available")
    explicit = cast(ExplicitBackendName, backend)
    available, reason = probe_backend(explicit)
    if not available:
        raise RuntimeError(reason)
    return explicit


def _array_from_mapping(result: Mapping[str, object], key: str) -> FloatArray:
    try:
        value = result[key]
    except KeyError as exc:
        raise RuntimeError(f"native backend result is missing {key!r}") from exc
    try:
        return np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(f"native backend returned a non-numeric {key!r}") from exc


def _float_from_object(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise RuntimeError(f"native backend returned a non-numeric {name}")
    try:
        return float(value)
    except OverflowError as exc:
        raise RuntimeError(
            f"native backend returned an out-of-range {name}",
        ) from exc


def _validated_native_result(
    *,
    backend: str,
    time_steps: int,
    state_dim: int,
    means: object,
    covariances: object,
    pred_means: object,
    pred_covariances: object,
    log_likelihood: object,
) -> FilterResult:
    try:
        result = FilterResult(
            means=np.asarray(means, dtype=np.float64),
            covariances=np.asarray(covariances, dtype=np.float64),
            pred_means=np.asarray(pred_means, dtype=np.float64),
            pred_covariances=np.asarray(pred_covariances, dtype=np.float64),
            log_likelihood=_float_from_object(
                log_likelihood,
                name="log_likelihood",
            ),
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{backend} backend returned invalid filter moments",
        ) from exc
    expected_shape = (time_steps, state_dim)
    if result.means.shape != expected_shape:
        raise RuntimeError(
            f"{backend} backend returned means shape {result.means.shape}; "
            f"expected {expected_shape}",
        )
    return result


def _filter_rust(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> FilterResult:
    if _rust_kalman_filter is None:
        raise RuntimeError("Rust backend was selected without a loaded filter")
    time_steps, obs_dim = observations.shape
    raw_result = _rust_kalman_filter(
        obs_flat=observations.ravel(order="C").tolist(),
        controls_flat=controls.ravel(order="C").tolist(),
        t_len=time_steps,
        p_dim=obs_dim,
        m_dim=model.control_dim,
        a_flat=model.A.ravel(order="C").tolist(),
        b_flat=model.B.ravel(order="C").tolist(),
        c_flat=model.C.ravel(order="C").tolist(),
        d_flat=model.D.ravel(order="C").tolist(),
        q_flat=model.Q.ravel(order="C").tolist(),
        r_flat=model.R.ravel(order="C").tolist(),
        mu_0=model.mu_0.tolist(),
        sigma_0_flat=model.Sigma_0.ravel(order="C").tolist(),
        d_dim=model.state_dim,
    )
    if not isinstance(raw_result, Mapping):
        raise RuntimeError("Rust backend returned a non-mapping filter result")
    result = cast(Mapping[str, object], raw_result)
    return _validated_native_result(
        backend="Rust",
        time_steps=time_steps,
        state_dim=model.state_dim,
        means=_array_from_mapping(result, "means"),
        covariances=_array_from_mapping(result, "covariances"),
        pred_means=_array_from_mapping(result, "pred_means"),
        pred_covariances=_array_from_mapping(result, "pred_covariances"),
        log_likelihood=result.get("log_likelihood"),
    )


def _filter_julia(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> FilterResult:
    if _julia_module is None:
        raise RuntimeError("Julia backend was selected without a loaded module")
    try:
        result = _julia_module.kalman_filter(
            observations,
            controls,
            model.A,
            model.B,
            model.C,
            model.D,
            model.Q,
            model.R,
            model.mu_0,
            model.Sigma_0,
        )
        return _validated_native_result(
            backend="Julia",
            time_steps=observations.shape[0],
            state_dim=model.state_dim,
            means=result.means,
            covariances=result.covariances,
            pred_means=result.pred_means,
            pred_covariances=result.pred_covs,
            log_likelihood=result.log_lik,
        )
    except AttributeError as exc:
        raise RuntimeError("Julia backend returned an incomplete filter result") from exc


def _empty_output_buffers(
    time_steps: int,
    state_dim: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    return (
        np.zeros((time_steps, state_dim), dtype=np.float64),
        np.zeros((time_steps, state_dim, state_dim), dtype=np.float64),
        np.zeros((time_steps, state_dim), dtype=np.float64),
        np.zeros((time_steps, state_dim, state_dim), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )


def _filter_go(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> FilterResult:
    if _go_lib is None:
        raise RuntimeError("Go backend was selected without a loaded library")
    time_steps, obs_dim = observations.shape
    state_dim = model.state_dim
    arrays = [observations, controls, model.A, model.B, model.C, model.D, model.Q, model.R]
    arrays.extend((model.mu_0, model.Sigma_0))
    outputs = _empty_output_buffers(time_steps, state_dim)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    input_pointers = [array.ctypes.data_as(double_pointer) for array in arrays]
    output_pointers = [array.ctypes.data_as(double_pointer) for array in outputs]
    _go_lib.kalman_filter_c(
        *input_pointers,
        ctypes.c_int(time_steps),
        ctypes.c_int(obs_dim),
        ctypes.c_int(model.control_dim),
        ctypes.c_int(state_dim),
        *output_pointers,
    )
    return _validated_native_result(
        backend="Go",
        time_steps=time_steps,
        state_dim=state_dim,
        means=outputs[0],
        covariances=outputs[1],
        pred_means=outputs[2],
        pred_covariances=outputs[3],
        log_likelihood=float(outputs[4][0]),
    )


def _filter_mojo(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> FilterResult:
    if _mojo_lib is None:
        raise RuntimeError("Mojo backend was selected without a loaded library")
    time_steps, obs_dim = observations.shape
    state_dim = model.state_dim
    arrays = [observations, controls, model.A, model.B, model.C, model.D, model.Q, model.R]
    arrays.extend((model.mu_0, model.Sigma_0))
    outputs = _empty_output_buffers(time_steps, state_dim)
    _mojo_lib.kalman_filter_c(
        *(array.ctypes.data for array in arrays),
        time_steps,
        obs_dim,
        model.control_dim,
        state_dim,
        *(array.ctypes.data for array in outputs),
    )
    return _validated_native_result(
        backend="Mojo",
        time_steps=time_steps,
        state_dim=state_dim,
        means=outputs[0],
        covariances=outputs[1],
        pred_means=outputs[2],
        pred_covariances=outputs[3],
        log_likelihood=float(outputs[4][0]),
    )


def filter_native(
    backend: ExplicitBackendName,
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> FilterResult:
    """Execute a resolved non-Python backend with validated float64 buffers."""
    runners = {
        "rust": _filter_rust,
        "julia": _filter_julia,
        "go": _filter_go,
        "mojo": _filter_mojo,
    }
    if backend == "python":
        raise ValueError("filter_native cannot execute the Python backend")
    return runners[backend](model, observations, controls)
