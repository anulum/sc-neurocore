# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts acceleration loading and result validation

"""Load exact Rust, Julia, Go and Mojo McCulloch--Pitts batch lanes."""

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

McCullochPittsResult = tuple[npt.NDArray[np.uint8], int]
KERNEL = "mcculloch_pitts_absolute_inhibition_batch"
_AUTO_BACKENDS = with_floor("python")


class _EngineRunner(Protocol):
    """Typed PyO3 batch boundary."""

    def __call__(
        self,
        theta: int,
        excitatory_counts: npt.NDArray[np.int64],
        inhibitory_flags: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.uint8], int]: ...


def _load_engine_runner() -> _EngineRunner:
    """Return the installed Rust engine's full-contract batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_mcculloch_pitts_evaluate_batch)


try:
    _engine_evaluate: _EngineRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_evaluate = None
    _HAS_RUST = False

_julia_module: Any | None = None
_go_lib: Any | None = None
_mojo_lib: Any | None = None
_HAS_JULIA = False
_HAS_GO = False
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the committed Julia module when ``juliacall`` is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source = os.path.join(_ACCEL_ROOT, "julia", "neurons", "mcculloch_pitts.jl")
    if not os.path.isfile(source):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source)
        _julia_module = julia.McCullochPittsAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, *, mojo: bool) -> Any | None:
    """Bind one exact batch C ABI."""
    evaluate = getattr(library, "mcculloch_pitts_evaluate_c", None)
    if evaluate is None:
        return None
    if mojo:
        evaluate.argtypes = [ctypes.c_int64] * 5
    else:
        evaluate.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint8),
        ]
    evaluate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the staged Go C-shared McCulloch--Pitts library."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(
        _ACCEL_ROOT,
        "go",
        "neurons",
        "mcculloch_pitts",
        "libmcculloch_pitts.so",
    )
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
    """Load the staged Mojo McCulloch--Pitts shared library."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libmcculloch_pitts.so")
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
        return _HAS_RUST and _engine_evaluate is not None
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


def normalise_result(result: object, *, expected_length: int) -> McCullochPittsResult:
    """Reject malformed native output before returning the public binary trace."""
    if isinstance(result, tuple) and len(result) == 2:
        events, event_count = result
    elif hasattr(result, "events") and hasattr(result, "event_count"):
        events = result.events
        event_count = result.event_count
    else:
        raise FloatingPointError("McCulloch-Pitts backend returned a malformed result.")
    try:
        values = np.asarray(events, dtype=np.float64)
        count_float = float(cast(SupportsFloat, event_count))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("McCulloch-Pitts backend returned non-numeric output.") from exc
    if values.ndim != 1 or values.shape != (expected_length,):
        raise FloatingPointError("McCulloch-Pitts backend returned a malformed event trace.")
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise FloatingPointError("McCulloch-Pitts backend returned non-binary events.")
    if (
        isinstance(event_count, (bool, np.bool_))
        or not math_is_finite_integer(count_float)
        or not 0 <= count_float <= expected_length
    ):
        raise FloatingPointError("McCulloch-Pitts backend returned an invalid event count.")
    count = int(count_float)
    normalised = np.ascontiguousarray(values, dtype=np.uint8)
    if count != int(np.sum(normalised, dtype=np.int64)):
        raise FloatingPointError(
            "McCulloch-Pitts backend event count disagrees with its event trace."
        )
    return normalised, count


def math_is_finite_integer(value: float) -> bool:
    """Return whether a converted backend scalar is finite and integral."""
    return bool(np.isfinite(value) and value.is_integer())


def evaluate_rust(
    theta: int,
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.uint8],
) -> McCullochPittsResult:
    """Evaluate the complete batch through the production Rust engine."""
    if _engine_evaluate is None:
        raise RuntimeError("Rust McCulloch-Pitts engine is unavailable.")
    return normalise_result(
        _engine_evaluate(theta, counts, flags),
        expected_length=len(counts),
    )


def evaluate_julia(
    theta: int,
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.uint8],
) -> McCullochPittsResult:
    """Evaluate the complete batch through the committed Julia module."""
    if _julia_module is None:
        raise RuntimeError("Julia McCulloch-Pitts module is unavailable.")
    result = _julia_module.evaluate_batch(theta, counts, flags)
    return normalise_result(result, expected_length=len(counts))


def _evaluate_c(
    library: Any,
    theta: int,
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.uint8],
    *,
    mojo: bool,
) -> McCullochPittsResult:
    """Evaluate one staged C ABI after allocating its complete destination."""
    output = np.empty(len(counts), dtype=np.uint8)
    if mojo:
        event_count = library.mcculloch_pitts_evaluate_c(
            theta,
            int(counts.ctypes.data),
            int(flags.ctypes.data),
            len(counts),
            int(output.ctypes.data),
        )
    else:
        event_count = library.mcculloch_pitts_evaluate_c(
            theta,
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(counts),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    if int(event_count) < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} McCulloch-Pitts kernel rejected the contract.")
    return normalise_result((output, event_count), expected_length=len(counts))


def evaluate_go(
    theta: int,
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.uint8],
) -> McCullochPittsResult:
    """Evaluate the Go recurrence through its generated C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go McCulloch-Pitts library is unavailable.")
    return _evaluate_c(_go_lib, theta, counts, flags, mojo=False)


def evaluate_mojo(
    theta: int,
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.uint8],
) -> McCullochPittsResult:
    """Evaluate the Mojo recurrence through its shared-library ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo McCulloch-Pitts library is unavailable.")
    return _evaluate_c(_mojo_lib, theta, counts, flags, mojo=True)


__all__ = [
    "KERNEL",
    "auto_backend",
    "backend_available",
    "ensure_go_loaded",
    "ensure_julia_loaded",
    "ensure_mojo_loaded",
    "evaluate_go",
    "evaluate_julia",
    "evaluate_mojo",
    "evaluate_rust",
    "normalise_result",
]
