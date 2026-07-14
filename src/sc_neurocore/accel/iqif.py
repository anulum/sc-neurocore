# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF acceleration backend loading and exact result validation

"""Load full-contract Rust, Julia, Go and Mojo IQIF batch backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, Protocol, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

IQIFResult = tuple[npt.NDArray[np.int64], int, int]
KERNEL = "iqif_integer_q03_batch"
_AUTO_BACKENDS = with_floor("python")


class _EngineRunner(Protocol):
    """Typed PyO3 IQIF batch boundary."""

    def __call__(
        self,
        v: int,
        v_rest: int,
        v_threshold: int,
        v_reset: int,
        a: int,
        b: int,
        v_max: int,
        v_min: int,
        n_steps: int,
        current: int,
    ) -> tuple[npt.NDArray[np.int64], int, int]: ...


def _load_engine_iqif() -> _EngineRunner:
    """Return the installed Rust engine's exact IQIF batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_iqif_simulate)


try:
    _engine_simulate: _EngineRunner | None = _load_engine_iqif()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False

_julia_module: Any | None = None
_go_lib: Any | None = None
_mojo_lib: Any | None = None
_HAS_JULIA = False
_HAS_GO = False
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)
_INTEGER_ARGUMENTS = 10


def ensure_julia_loaded() -> bool:
    """Load the committed Julia IQIF module when ``juliacall`` is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source = os.path.join(_ACCEL_ROOT, "julia", "neurons", "iqif.jl")
    if not os.path.isfile(source):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source)
        _julia_module = julia.IQIFAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, *, mojo: bool) -> Any | None:
    """Bind one exact IQIF C ABI."""
    simulate = getattr(library, "iqif_simulate_c", None)
    if simulate is None:
        return None
    destination = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [ctypes.c_int64] * _INTEGER_ARGUMENTS + [destination]
    simulate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the staged Go IQIF C-shared library."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "go", "neurons", "iqif", "libiqif.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _go_lib = _configure_c_library(library, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def ensure_mojo_loaded() -> bool:
    """Load the staged Mojo IQIF shared library."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libiqif.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def backend_available(backend: str) -> bool:
    """Return whether one public execution lane is ready."""
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        return ensure_julia_loaded()
    if backend == "go":
        return ensure_go_loaded()
    if backend == "mojo":
        return ensure_mojo_loaded()
    return backend == "python"


def auto_backend() -> str:
    """Choose the first available lane from committed measured evidence."""
    ordered = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in ordered if backend_available(backend)), "python")


def normalise_result(
    trace: npt.ArrayLike,
    spikes: object,
    final_v: object,
    *,
    n_steps: int,
    v_min: int,
    v_max: int,
) -> IQIFResult:
    """Reject malformed or lossy backend output before narrowing to int64."""
    try:
        values = np.asarray(trace, dtype=np.float64)
        final_float = float(cast(SupportsFloat, final_v))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("IQIF backend returned non-numeric state.") from exc
    if values.ndim != 1 or values.shape != (n_steps,):
        raise FloatingPointError("IQIF backend returned a malformed voltage trace.")
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.trunc(values)).all()
        or not np.logical_and(values >= v_min, values <= v_max).all()
    ):
        raise FloatingPointError("IQIF backend returned non-integral or out-of-range voltage.")
    if (
        isinstance(final_v, bool)
        or not np.isfinite(final_float)
        or not final_float.is_integer()
        or not v_min <= final_float <= v_max
    ):
        raise FloatingPointError("IQIF backend returned an invalid final voltage.")
    if isinstance(spikes, bool):
        raise FloatingPointError("IQIF backend returned an invalid spike count.")
    try:
        spike_float = float(cast(SupportsFloat, spikes))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("IQIF backend returned an invalid spike count.") from exc
    if not np.isfinite(spike_float) or not spike_float.is_integer():
        raise FloatingPointError("IQIF backend returned an invalid spike count.")
    spike_count = int(spike_float)
    if not 0 <= spike_count <= n_steps:
        raise FloatingPointError("IQIF backend returned an invalid spike count.")
    final_int = int(final_float)
    if n_steps and int(values[-1]) != final_int:
        raise FloatingPointError("IQIF backend final voltage disagrees with its trace.")
    return np.ascontiguousarray(values, dtype=np.int64), spike_count, final_int


def simulate_rust(
    v: int,
    v_rest: int,
    v_threshold: int,
    v_reset: int,
    a: int,
    b: int,
    v_max: int,
    v_min: int,
    n_steps: int,
    current: int,
) -> IQIFResult:
    """Run the complete contract through the production Rust engine."""
    if _engine_simulate is None:
        raise RuntimeError("Rust IQIF engine is unavailable.")
    return normalise_result(
        *_engine_simulate(
            v,
            v_rest,
            v_threshold,
            v_reset,
            a,
            b,
            v_max,
            v_min,
            n_steps,
            current,
        ),
        n_steps=n_steps,
        v_min=v_min,
        v_max=v_max,
    )


def simulate_julia(
    v: int,
    v_rest: int,
    v_threshold: int,
    v_reset: int,
    a: int,
    b: int,
    v_max: int,
    v_min: int,
    n_steps: int,
    current: int,
) -> IQIFResult:
    """Run the complete contract through the committed Julia module."""
    if _julia_module is None:
        raise RuntimeError("Julia IQIF module is unavailable.")
    result = _julia_module.simulate_trace(
        v,
        v_rest,
        v_threshold,
        v_reset,
        a,
        b,
        v_max,
        v_min,
        n_steps,
        current,
    )
    return normalise_result(
        result.trace,
        result.spikes,
        result.vf,
        n_steps=n_steps,
        v_min=v_min,
        v_max=v_max,
    )


def _simulate_c(
    library: Any,
    v: int,
    v_rest: int,
    v_threshold: int,
    v_reset: int,
    a: int,
    b: int,
    v_max: int,
    v_min: int,
    n_steps: int,
    current: int,
    *,
    mojo: bool,
) -> IQIFResult:
    """Run one staged C ABI and validate its Float64 integer transport."""
    output = np.empty(n_steps + 1, dtype=np.float64)
    destination: Any
    if mojo:
        destination = int(output.ctypes.data)
    else:
        destination = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    spikes = int(
        library.iqif_simulate_c(
            v,
            v_rest,
            v_threshold,
            v_reset,
            a,
            b,
            v_max,
            v_min,
            n_steps,
            current,
            destination,
        )
    )
    if spikes < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} IQIF kernel rejected the contract.")
    return normalise_result(
        output[:n_steps],
        spikes,
        output[n_steps],
        n_steps=n_steps,
        v_min=v_min,
        v_max=v_max,
    )


def simulate_go(
    v: int,
    v_rest: int,
    v_threshold: int,
    v_reset: int,
    a: int,
    b: int,
    v_max: int,
    v_min: int,
    n_steps: int,
    current: int,
) -> IQIFResult:
    """Run the Go recurrence through its generated C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go IQIF library is unavailable.")
    return _simulate_c(
        _go_lib,
        v,
        v_rest,
        v_threshold,
        v_reset,
        a,
        b,
        v_max,
        v_min,
        n_steps,
        current,
        mojo=False,
    )


def simulate_mojo(
    v: int,
    v_rest: int,
    v_threshold: int,
    v_reset: int,
    a: int,
    b: int,
    v_max: int,
    v_min: int,
    n_steps: int,
    current: int,
) -> IQIFResult:
    """Run the Mojo recurrence through its shared-library ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo IQIF library is unavailable.")
    return _simulate_c(
        _mojo_lib,
        v,
        v_rest,
        v_threshold,
        v_reset,
        a,
        b,
        v_max,
        v_min,
        n_steps,
        current,
        mojo=True,
    )
