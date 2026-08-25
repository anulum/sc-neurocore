# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — parity-checked SC two-state chaotic-map dispatch

"""Dispatch and validate complete batches of the project SC chaotic map."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.sc_chaotic_map_neuron import (
    SCChaoticMapNeuron,
    SCChaoticMapResult,
)

KERNEL = "sc_chaotic_map_batch"
PARITY_ATOL = {"python": 0.0, "rust": 1e-12, "julia": 1e-12, "go": 1e-12, "mojo": 1e-10}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RECEIPT_ATOL = 1e-10


class _BatchRunner(Protocol):
    def __call__(
        self,
        x: float,
        y: float,
        k_f: float,
        k_s: float,
        alpha: float,
        delta: float,
        x_threshold: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_sc_chaotic_map_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_chaotic_map")


def backend_available(backend: str) -> bool:
    """Return whether one executable SC-map lane is available."""
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            module = importlib.import_module("sc_neurocore.accel.julia.neurons.sc_chaotic_map")
            module._ensure_loaded()
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
        return bool(getattr(module, f"_HAS_{backend.upper()}_SC_CHAOTIC_MAP", False))
    return False


def auto_backend() -> str:
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


def _unit(
    x: float,
    y: float,
    k_f: float,
    k_s: float,
    alpha: float,
    delta: float,
    x_threshold: float,
) -> SCChaoticMapNeuron:
    return SCChaoticMapNeuron(x, y, k_f, k_s, alpha, delta, x_threshold)


def normalise_result(
    result: dict[str, object], *, n_steps: int, initial_x: float, initial_y: float, threshold: float
) -> SCChaoticMapResult:
    """Validate complete traces, edge events, and final receipts."""
    normalised: SCChaoticMapResult = {}
    for key in ("x", "y", "spikes"):
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(
                f"SC chaotic-map backend returned invalid {key} trace"
            ) from exc
        if values.ndim != 1 or values.shape != (n_steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SC chaotic-map backend returned malformed {key} trace")
        normalised[key] = np.ascontiguousarray(values)

    x_trace = cast(npt.NDArray[np.float64], normalised["x"])
    y_trace = cast(npt.NDArray[np.float64], normalised["y"])
    spikes = cast(npt.NDArray[np.float64], normalised["spikes"])
    if not np.all((x_trace >= -10.0) & (x_trace <= 10.0)) or not np.all(
        (y_trace >= -10.0) & (y_trace <= 10.0)
    ):
        raise FloatingPointError("SC chaotic-map backend violated the state clamp")
    if not np.isin(spikes, (0.0, 1.0)).all():
        raise FloatingPointError("SC chaotic-map backend returned non-binary events")
    previous = np.concatenate((np.asarray([initial_x]), x_trace[:-1]))
    expected_events = ((previous < threshold) & (x_trace >= threshold)).astype(np.float64)
    if not np.array_equal(spikes, expected_events):
        raise FloatingPointError("SC chaotic-map backend violated upward-crossing semantics")

    for key, expected in (
        ("x_final", initial_x if n_steps == 0 else float(x_trace[-1])),
        ("y_final", initial_y if n_steps == 0 else float(y_trace[-1])),
    ):
        try:
            value = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"SC chaotic-map backend returned invalid {key}") from exc
        if not np.isfinite(value) or abs(value - expected) > _RECEIPT_ATOL:
            raise FloatingPointError(f"SC chaotic-map {key} disagrees with its trace")
        normalised[key] = value

    raw_count = result.get("spike_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
        raise FloatingPointError("SC chaotic-map backend returned invalid spike_count")
    count = int(raw_count)
    if count != int(np.sum(spikes, dtype=np.float64)):
        raise FloatingPointError("SC chaotic-map spike_count disagrees with its trace")
    normalised["spike_count"] = count
    return normalised


def simulate_python(
    x: float,
    y: float,
    k_f: float,
    k_s: float,
    alpha: float,
    delta: float,
    x_threshold: float,
    current: npt.ArrayLike,
) -> SCChaoticMapResult:
    unit = _unit(x, y, k_f, k_s, alpha, delta, x_threshold)
    drive = _input(current)
    x_trace = np.empty(drive.size, dtype=np.float64)
    y_trace = np.empty(drive.size, dtype=np.float64)
    spikes = np.empty(drive.size, dtype=np.float64)
    count = 0
    for index, value in enumerate(drive):
        event = unit.step(float(value))
        x_trace[index], y_trace[index], spikes[index] = unit.x, unit.y, event
        count += event
    return normalise_result(
        {
            "x": x_trace,
            "y": y_trace,
            "spikes": spikes,
            "x_final": unit.x,
            "y_final": unit.y,
            "spike_count": count,
        },
        n_steps=drive.size,
        initial_x=x,
        initial_y=y,
        threshold=x_threshold,
    )


def _simulate_julia(*args: object) -> dict[str, object]:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.sc_chaotic_map")
    return cast(dict[str, object], module.simulate_sc_chaotic_map(*args))


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust SC chaotic-map backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return cast(_BatchRunner, _simulate_julia)
    return cast(_BatchRunner, _native_module(backend).simulate_sc_chaotic_map)


def simulate_sc_chaotic_map(
    x: float = 0.0,
    y: float = 0.0,
    k_f: float = 0.7,
    k_s: float = 0.95,
    alpha: float = 2.0,
    delta: float = 0.05,
    x_threshold: float = 0.5,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> SCChaoticMapResult:
    unit = _unit(x, y, k_f, k_s, alpha, delta, x_threshold)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown SC chaotic-map backend: {selected}")
    args = (unit.x, unit.y, unit.k_f, unit.k_s, unit.alpha, unit.delta, unit.x_threshold, drive)
    if selected == "python":
        return simulate_python(*args)
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} SC chaotic-map backend is unavailable")
    return normalise_result(
        _native_runner(selected)(*args),
        n_steps=drive.size,
        initial_x=unit.x,
        initial_y=unit.y,
        threshold=unit.x_threshold,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_python",
    "simulate_sc_chaotic_map",
]
