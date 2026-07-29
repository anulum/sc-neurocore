# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — parity-checked SC adaptive-threshold-map dispatch

"""Dispatch and validate complete batches of the retained SC project map."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.sc_adaptive_threshold_map_neuron import (
    SCAdaptiveThresholdMapNeuron,
    SCAdaptiveThresholdMapResult,
)

KERNEL = "sc_adaptive_threshold_map_batch"
PARITY_ATOL = {"python": 0.0, "rust": 1e-12, "julia": 1e-12, "go": 1e-12, "mojo": 1e-10}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RECEIPT_ATOL = 1e-10


class _BatchRunner(Protocol):
    def __call__(
        self,
        x: float,
        theta: float,
        k: float,
        beta: float,
        gamma: float,
        theta_spike: float,
        x_threshold: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_sc_adaptive_threshold_map_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_adaptive_threshold_map")


def backend_available(backend: str) -> bool:
    """Return whether one executable SC adaptive-map lane is ready."""
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            module = importlib.import_module(
                "sc_neurocore.accel.julia.neurons.sc_adaptive_threshold_map"
            )
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
        flag = f"_HAS_{backend.upper()}_SC_ADAPTIVE_THRESHOLD_MAP"
        return bool(getattr(module, flag, False))
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


def _unit(*args: float) -> SCAdaptiveThresholdMapNeuron:
    return SCAdaptiveThresholdMapNeuron(*args)


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial_x: float,
    initial_theta: float,
    threshold: float,
) -> SCAdaptiveThresholdMapResult:
    """Validate state traces, upward events, and scalar receipts."""
    normalised: SCAdaptiveThresholdMapResult = {}
    for key in ("x", "theta", "spikes"):
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"SC adaptive-map backend returned invalid {key}") from exc
        if values.ndim != 1 or values.shape != (n_steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SC adaptive-map backend returned malformed {key}")
        normalised[key] = np.ascontiguousarray(values)
    x_trace = cast(npt.NDArray[np.float64], normalised["x"])
    theta_trace = cast(npt.NDArray[np.float64], normalised["theta"])
    spikes = cast(npt.NDArray[np.float64], normalised["spikes"])
    if not np.all((x_trace >= -5.0) & (x_trace <= 5.0)) or not np.all(
        (theta_trace >= -5.0) & (theta_trace <= 5.0)
    ):
        raise FloatingPointError("SC adaptive-map backend violated its state clamp")
    if not np.isin(spikes, (0.0, 1.0)).all():
        raise FloatingPointError("SC adaptive-map backend returned non-binary events")
    previous = np.concatenate((np.asarray([initial_x]), x_trace[:-1]))
    expected_events = ((previous < threshold) & (x_trace >= threshold)).astype(np.float64)
    if not np.array_equal(spikes, expected_events):
        raise FloatingPointError("SC adaptive-map backend violated upward-crossing semantics")
    for key, expected in (
        ("x_final", initial_x if n_steps == 0 else float(x_trace[-1])),
        ("theta_final", initial_theta if n_steps == 0 else float(theta_trace[-1])),
    ):
        try:
            value = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"SC adaptive-map backend returned invalid {key}") from exc
        if not np.isfinite(value) or abs(value - expected) > _RECEIPT_ATOL:
            raise FloatingPointError(f"SC adaptive-map {key} disagrees with its trace")
        normalised[key] = value
    raw_count = result.get("spike_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
        raise FloatingPointError("SC adaptive-map backend returned invalid spike_count")
    count = int(raw_count)
    if count != int(np.sum(spikes, dtype=np.float64)):
        raise FloatingPointError("SC adaptive-map spike_count disagrees with its trace")
    normalised["spike_count"] = count
    return normalised


def simulate_python(
    x: float,
    theta: float,
    k: float,
    beta: float,
    gamma: float,
    theta_spike: float,
    x_threshold: float,
    current: npt.ArrayLike,
) -> SCAdaptiveThresholdMapResult:
    """Run a complete batch through the Python project specification."""
    unit = _unit(x, theta, k, beta, gamma, theta_spike, x_threshold)
    drive = _input(current)
    x_trace = np.empty(drive.size, dtype=np.float64)
    theta_trace = np.empty(drive.size, dtype=np.float64)
    spikes = np.empty(drive.size, dtype=np.float64)
    count = 0
    for index, value in enumerate(drive):
        event = unit.step(float(value))
        x_trace[index], theta_trace[index], spikes[index] = unit.x, unit.theta, event
        count += event
    return normalise_result(
        {
            "x": x_trace,
            "theta": theta_trace,
            "spikes": spikes,
            "x_final": unit.x,
            "theta_final": unit.theta,
            "spike_count": count,
        },
        n_steps=drive.size,
        initial_x=x,
        initial_theta=theta,
        threshold=x_threshold,
    )


def _simulate_julia(*args: object) -> dict[str, object]:
    module = importlib.import_module("sc_neurocore.accel.julia.neurons.sc_adaptive_threshold_map")
    return cast(dict[str, object], module.simulate_sc_adaptive_threshold_map(*args))


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust SC adaptive-map backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return cast(_BatchRunner, _simulate_julia)
    return cast(_BatchRunner, _native_module(backend).simulate_sc_adaptive_threshold_map)


def simulate_sc_adaptive_threshold_map(
    x: float = 0.0,
    theta: float = 0.0,
    k: float = 1.5,
    beta: float = 0.95,
    gamma: float = 0.3,
    theta_spike: float = 0.8,
    x_threshold: float = 0.8,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> SCAdaptiveThresholdMapResult:
    """Run one complete retained-project-model batch on a selected lane."""
    unit = _unit(x, theta, k, beta, gamma, theta_spike, x_threshold)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown SC adaptive-map backend: {selected}")
    args = (
        unit.x,
        unit.theta,
        unit.k,
        unit.beta,
        unit.gamma,
        unit.theta_spike,
        unit.x_threshold,
        drive,
    )
    if selected == "python":
        return simulate_python(*args)
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} SC adaptive-map backend is unavailable")
    return normalise_result(
        _native_runner(selected)(*args),
        n_steps=drive.size,
        initial_x=unit.x,
        initial_theta=unit.theta,
        threshold=unit.x_threshold,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_python",
    "simulate_sc_adaptive_threshold_map",
]
