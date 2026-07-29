# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-faithful Nagumo–Sato accelerator dispatch

"""Dispatch and validate complete Nagumo–Sato state and firing traces."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.nagumo_sato_map_neuron import (
    NAGUMO_SATO_INITIAL_Y,
    NagumoSatoMapNeuron,
    NagumoSatoMapResult,
)

KERNEL = "nagumo_sato_map_batch"
PARITY_ATOL = {"python": 0.0, "rust": 0.0, "julia": 0.0, "go": 0.0, "mojo": 5e-15}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RECEIPT_ATOL = 1.0e-12


class _BatchRunner(Protocol):
    def __call__(
        self,
        y: float,
        k: float,
        alpha: float,
        bias: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_nagumo_sato_map_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.nagumo_sato_map")


def _ensure_julia_loaded() -> Any:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.nagumo_sato_map")
    return module._ensure_loaded()


def backend_available(backend: str) -> bool:
    """Return whether one maintained execution lane is ready."""
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            _ensure_julia_loaded()
        except (ImportError, FileNotFoundError):
            return False
        except Exception as exc:
            if exc.__class__.__name__ == "JuliaError":
                return False
            raise
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_NAGUMO_SATO_MAP", False))
    return False


def auto_backend() -> str:
    """Return the first available lane in measured ascending-latency order."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _input(current: npt.ArrayLike) -> npt.NDArray[np.float64]:
    logical = np.asarray(current)
    if logical.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {logical.shape}")
    if logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"current exceeds the signed-32-bit step limit: {logical.size}")
    values = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("current must contain only finite values")
    return values


def _unit(y: float, k: float, alpha: float, bias: float) -> NagumoSatoMapNeuron:
    return NagumoSatoMapNeuron(y=y, k=k, alpha=alpha, bias=bias)


def normalise_result(
    result: dict[str, object], *, n_steps: int, initial_y: float
) -> NagumoSatoMapResult:
    """Validate complete state/output traces and scalar receipts."""
    normalised: NagumoSatoMapResult = {}
    for key in ("y", "x", "spikes"):
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Nagumo-Sato backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"Nagumo-Sato backend returned malformed {key} trace")
        normalised[key] = np.ascontiguousarray(values)

    y_trace = cast(npt.NDArray[np.float64], normalised["y"])
    output = cast(npt.NDArray[np.float64], normalised["x"])
    spikes = cast(npt.NDArray[np.float64], normalised["spikes"])
    expected = (y_trace >= 0.0).astype(np.float64)
    if not np.array_equal(output, expected) or not np.array_equal(spikes, expected):
        raise FloatingPointError("Nagumo-Sato output disagrees with H(y[t+1])")
    expected_y = initial_y if n_steps == 0 else float(y_trace[-1])
    expected_x = float(initial_y >= 0.0) if n_steps == 0 else float(output[-1])
    for key, expected_value in (("y_final", expected_y), ("x_final", expected_x)):
        try:
            value = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Nagumo-Sato backend returned invalid {key}") from exc
        if not math_isfinite(value) or abs(value - expected_value) > _RECEIPT_ATOL:
            raise FloatingPointError(f"Nagumo-Sato {key} disagrees with its trace")
        normalised[key] = value
    raw_count = result.get("spike_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
        raise FloatingPointError("Nagumo-Sato backend returned invalid spike_count")
    count = int(raw_count)
    if count != int(np.sum(spikes, dtype=np.float64)):
        raise FloatingPointError("Nagumo-Sato spike_count disagrees with its trace")
    normalised["spike_count"] = count
    return normalised


def math_isfinite(value: float) -> bool:
    """Avoid importing a scalar math helper into the public surface."""
    return bool(np.isfinite(value))


def simulate_python(
    y: float,
    k: float,
    alpha: float,
    bias: float,
    current: npt.ArrayLike,
) -> NagumoSatoMapResult:
    """Run a complete batch through the Python golden model."""
    unit = _unit(y, k, alpha, bias)
    drive = _input(current)
    y_trace = np.empty(drive.size, dtype=np.float64)
    output = np.empty(drive.size, dtype=np.float64)
    count = 0
    for index, value in enumerate(drive):
        event = unit.step(float(value))
        y_trace[index], output[index] = unit.y, event
        count += event
    return normalise_result(
        {
            "y": y_trace,
            "x": output,
            "spikes": output.copy(),
            "y_final": unit.y,
            "x_final": unit.output(),
            "spike_count": count,
        },
        n_steps=drive.size,
        initial_y=y,
    )


def _simulate_julia(*args: object) -> dict[str, object]:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.nagumo_sato_map")
    return cast(dict[str, object], module.simulate_nagumo_sato_map(*args))


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust Nagumo-Sato backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return cast(_BatchRunner, _simulate_julia)
    return cast(_BatchRunner, _native_module(backend).simulate_nagumo_sato_map)


def simulate_nagumo_sato_map(
    y: float = NAGUMO_SATO_INITIAL_Y,
    k: float = 0.6,
    alpha: float = 1.0,
    bias: float = 0.2,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> NagumoSatoMapResult:
    """Run one complete source-faithful batch on a selected lane."""
    unit = _unit(y, k, alpha, bias)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown Nagumo-Sato backend: {selected}")
    arguments = (unit.y, unit.k, unit.alpha, unit.bias, drive)
    if selected == "python":
        return simulate_python(*arguments)
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} Nagumo-Sato backend is unavailable")
    return normalise_result(
        _native_runner(selected)(*arguments), n_steps=drive.size, initial_y=unit.y
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_nagumo_sato_map",
    "simulate_python",
]
