# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-faithful Aihara-map accelerator dispatch

"""Dispatch and validate the complete Aihara chaotic-neuron batch."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.aihara_map_neuron import (
    AIHARA_CHAOTIC_BIAS,
    AIHARA_INITIAL_Y,
    AiharaMapNeuron,
    AiharaMapResult,
)

KERNEL = "aihara_map_chaotic_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    # Mojo's libm exp differs at the last bits; chaotic amplification reaches
    # 1.7e-4 over the committed 512-step drive while events remain identical.
    "mojo": 2.0e-4,
}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RESULT_TOLERANCE = 1.0e-10
_TRACE_KEYS = ("y", "x", "spikes")


class _BatchRunner(Protocol):
    def __call__(
        self,
        y: float,
        k: float,
        alpha: float,
        bias: float,
        epsilon: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_aihara_map_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.aihara_map")


def _ensure_julia_loaded() -> Any:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.aihara_map")
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
        return bool(getattr(module, f"_HAS_{backend.upper()}_AIHARA_MAP", False))
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


def _unit(y: float, k: float, alpha: float, bias: float, epsilon: float) -> AiharaMapNeuron:
    return AiharaMapNeuron(y=y, k=k, alpha=alpha, bias=bias, epsilon=epsilon)


def _stable_logistic(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def normalise_result(
    result: dict[str, object], *, n_steps: int, initial_y: float, epsilon: float
) -> AiharaMapResult:
    """Validate complete state, graded-output, event traces, and receipts."""
    normalised: AiharaMapResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"aihara backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"aihara backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"aihara backend returned non-finite {key} trace")
        normalised[key] = np.ascontiguousarray(values)

    y_trace = cast(npt.NDArray[np.float64], normalised["y"])
    x_trace = cast(npt.NDArray[np.float64], normalised["x"])
    spike_trace = cast(npt.NDArray[np.float64], normalised["spikes"])
    if not np.all((x_trace >= 0.0) & (x_trace <= 1.0)):
        raise FloatingPointError("aihara backend returned an out-of-range logistic output")
    if not np.isin(spike_trace, (0.0, 1.0)).all():
        raise FloatingPointError("aihara backend returned a non-binary event trace")
    expected_x = _stable_logistic(y_trace / epsilon)
    if not np.allclose(x_trace, expected_x, rtol=0.0, atol=_RESULT_TOLERANCE):
        raise FloatingPointError("aihara output trace disagrees with the logistic read-out")
    if not np.array_equal(spike_trace, (x_trace >= 0.5).astype(np.float64)):
        raise FloatingPointError("aihara event trace disagrees with source waveform shaper")

    initial_x = AiharaMapNeuron._logistic(initial_y, epsilon)
    for key, expected in (
        ("y_final", initial_y if n_steps == 0 else float(y_trace[-1])),
        ("x_final", initial_x if n_steps == 0 else float(x_trace[-1])),
    ):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"aihara backend returned invalid {key}") from exc
        if not np.isfinite(final) or abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"aihara {key} disagrees with its trace")
        normalised[key] = final

    raw_count = result.get("spike_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
        raise FloatingPointError("aihara backend returned invalid spike_count")
    count = int(raw_count)
    if count != int(np.sum(spike_trace, dtype=np.float64)):
        raise FloatingPointError("aihara spike_count disagrees with its trace")
    normalised["spike_count"] = count
    return normalised


def simulate_python(
    y: float,
    k: float,
    alpha: float,
    bias: float,
    epsilon: float,
    current: npt.ArrayLike,
) -> AiharaMapResult:
    """Run a complete batch through the Python golden model."""
    unit = _unit(y, k, alpha, bias, epsilon)
    drive = _input(current)
    y_trace = np.empty(drive.size, dtype=np.float64)
    x_trace = np.empty(drive.size, dtype=np.float64)
    spikes = np.empty(drive.size, dtype=np.float64)
    count = 0
    for index, value in enumerate(drive):
        event = unit.step(float(value))
        y_trace[index] = unit.y
        x_trace[index] = unit.output()
        spikes[index] = event
        count += event
    return normalise_result(
        {
            "y": y_trace,
            "x": x_trace,
            "spikes": spikes,
            "y_final": unit.y,
            "x_final": unit.output(),
            "spike_count": count,
        },
        n_steps=drive.size,
        initial_y=y,
        epsilon=epsilon,
    )


def _simulate_julia(
    y: float,
    k: float,
    alpha: float,
    bias: float,
    epsilon: float,
    current: npt.NDArray[np.float64],
) -> dict[str, object]:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.aihara_map")
    return cast(dict[str, object], module.simulate_aihara_map(y, k, alpha, bias, epsilon, current))


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust aihara backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return _simulate_julia
    return cast(_BatchRunner, _native_module(backend).simulate_aihara_map)


def simulate_aihara_map(
    y: float = AIHARA_INITIAL_Y,
    k: float = 0.7,
    alpha: float = 1.0,
    bias: float = AIHARA_CHAOTIC_BIAS,
    epsilon: float = 0.01,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> AiharaMapResult:
    """Run one complete source-faithful batch on a selected execution lane."""
    unit = _unit(y, k, alpha, bias, epsilon)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown aihara backend: {selected}")
    arguments = (unit.y, unit.k, unit.alpha, unit.bias, unit.epsilon, drive)
    if selected == "python":
        return simulate_python(*arguments)
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} aihara backend is unavailable")
    return normalise_result(
        _native_runner(selected)(*arguments),
        n_steps=drive.size,
        initial_y=unit.y,
        epsilon=unit.epsilon,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_aihara_map",
    "simulate_python",
]
