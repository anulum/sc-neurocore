# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire measured-order accelerator dispatch

"""Dispatch and validate the complete exact-flow resonate-and-fire batch."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.resonate_and_fire import (
    ResonateAndFireNeuron,
    ResonateAndFireResult,
)

KERNEL = "resonate_and_fire_exact_flow_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "mojo": 1.0e-10,
}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RESULT_TOLERANCE = 1.0e-10
_TRACE_KEYS = ("x", "y", "spikes")
_FINAL_KEYS = ("x_final", "y_final")

_BatchArguments = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    npt.NDArray[np.float64],
]


class _BatchRunner(Protocol):
    def __call__(
        self,
        x: float,
        y: float,
        b: float,
        omega: float,
        threshold: float,
        dt: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_resonate_and_fire_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.resonate_and_fire")


def _ensure_julia_loaded() -> Any:
    """Load the kernel through the shared Julia neuron registry."""
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return registry._ensure_resonate_and_fire_loaded()


def backend_available(backend: str) -> bool:
    """Return whether one maintained execution lane is ready.

    Parameters
    ----------
    backend : str
        One of ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the named runtime and its Model40 artefact are available.
    """
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
        marker = f"_HAS_{backend.upper()}_RESONATE_AND_FIRE"
        return bool(getattr(module, marker, False))
    return False


def auto_backend() -> str:
    """Return the first available runtime in measured ascending-latency order.

    Returns
    -------
    str
        An available backend name, with ``python`` as the total fallback.
    """
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
    b: float,
    omega: float,
    threshold: float,
    dt: float,
) -> ResonateAndFireNeuron:
    return ResonateAndFireNeuron(
        x=x,
        y=y,
        b=b,
        omega=omega,
        threshold=threshold,
        dt=dt,
    )


def _arguments(
    unit: ResonateAndFireNeuron,
    current: npt.NDArray[np.float64],
) -> _BatchArguments:
    return (
        unit.x,
        unit.y,
        unit.b,
        unit.omega,
        unit.threshold,
        unit.dt,
        current,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float],
    threshold: float,
) -> ResonateAndFireResult:
    """Validate complete state/spike traces and scalar final receipts.

    Parameters
    ----------
    result : dict[str, object]
        Backend mapping with ``x``, ``y``, ``spikes``, both final states, and
        an integral ``spike_count``.
    n_steps : int
        Required length of every trajectory.
    initial : tuple[float, float]
        Initial ``x`` and ``y`` used to validate an empty batch.
    threshold : float
        Source voltage-coordinate threshold used to validate spike resets.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Contiguous finite trajectories and mutually consistent final receipts.

    Raises
    ------
    FloatingPointError
        If any backend field is missing, malformed, non-finite, inconsistent,
        non-binary, or violates the source reset contract.
    """
    normalised: ResonateAndFireResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(
                f"resonate-and-fire backend returned invalid {key} trace"
            ) from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"resonate-and-fire backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"resonate-and-fire backend returned non-finite {key} trace")
        if key == "spikes" and not np.isin(values, (0.0, 1.0)).all():
            raise FloatingPointError("resonate-and-fire backend returned a non-binary spike trace")
        normalised[key] = np.ascontiguousarray(values)

    for index, (key, state_key) in enumerate(zip(_FINAL_KEYS, ("x", "y"), strict=True)):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"resonate-and-fire backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"resonate-and-fire backend returned non-finite {key}")
        trace = cast(npt.NDArray[np.float64], normalised[state_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"resonate-and-fire {key} disagrees with its trace")
        normalised[key] = final

    try:
        raw_spike_count = result["spike_count"]
    except KeyError as exc:
        raise FloatingPointError("resonate-and-fire backend returned invalid spike_count") from exc
    if isinstance(raw_spike_count, bool) or not isinstance(raw_spike_count, Integral):
        raise FloatingPointError("resonate-and-fire backend returned invalid spike_count")
    spike_count = int(raw_spike_count)
    spike_trace = cast(npt.NDArray[np.float64], normalised["spikes"])
    if spike_count < 0 or spike_count != int(np.sum(spike_trace, dtype=np.float64)):
        raise FloatingPointError("resonate-and-fire spike_count disagrees with its trace")
    normalised["spike_count"] = spike_count

    x_trace = cast(npt.NDArray[np.float64], normalised["x"])
    y_trace = cast(npt.NDArray[np.float64], normalised["y"])
    if n_steps:
        reset_mask = spike_trace == 1.0
        if not np.all(x_trace[reset_mask] == 0.0):
            raise FloatingPointError("resonate-and-fire spike trace disagrees with x reset")
        if not np.all(y_trace[reset_mask] == threshold):
            raise FloatingPointError("resonate-and-fire spike trace disagrees with y reset")
    return normalised


def simulate_python(
    x: float,
    y: float,
    b: float,
    omega: float,
    threshold: float,
    dt: float,
    current: npt.ArrayLike,
) -> ResonateAndFireResult:
    """Run the complete batch through the Python golden model.

    Parameters
    ----------
    x, y : float
        Initial current-like and voltage-like state coordinates.
    b, omega, threshold, dt : float
        Complete exact-flow configuration.
    current : ArrayLike
        One finite real piecewise-constant current per maintained step.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Complete post-update traces, final states, and sampled crossing count.

    Raises
    ------
    ValueError
        If the configuration or current vector violates the public contract.
    FloatingPointError
        If an exact-flow candidate or returned receipt is non-finite.
    """
    unit = _unit(x, y, b, omega, threshold, dt)
    drive = _input(current)
    x_trace = np.empty(drive.size, dtype=np.float64)
    y_trace = np.empty(drive.size, dtype=np.float64)
    spikes = np.empty(drive.size, dtype=np.float64)
    spike_count = 0
    for index, value in enumerate(drive):
        spike = unit.step(float(value))
        x_trace[index] = unit.x
        y_trace[index] = unit.y
        spikes[index] = spike
        spike_count += spike
    result: dict[str, object] = {
        "x": x_trace,
        "y": y_trace,
        "spikes": spikes,
        "x_final": unit.x,
        "y_final": unit.y,
        "spike_count": spike_count,
    }
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(x, y),
        threshold=threshold,
    )


def _simulate_julia(
    x: float,
    y: float,
    b: float,
    omega: float,
    threshold: float,
    dt: float,
    current: npt.NDArray[np.float64],
) -> dict[str, object]:
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return cast(
        "dict[str, object]",
        registry.simulate_resonate_and_fire(
            x,
            y,
            b,
            omega,
            threshold,
            dt,
            current,
        ),
    )


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust resonate-and-fire backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return _simulate_julia
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_resonate_and_fire)


def simulate_resonate_and_fire(
    x: float = 0.0,
    y: float = 0.0,
    b: float = -1.0,
    omega: float = 10.0,
    threshold: float = 1.0,
    dt: float = 0.01,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> ResonateAndFireResult:
    """Run one complete exact-flow batch on a selected execution lane.

    Parameters
    ----------
    x, y : float, default: 0.0
        Initial current-like and voltage-like state coordinates.
    b : float, default: -1.0
        Radial damping or growth coefficient.
    omega : float, default: 10.0
        Positive angular resonance frequency.
    threshold : float, default: 1.0
        Positive spike threshold on the voltage-like ``y`` coordinate.
    dt : float, default: 0.01
        Positive piecewise-constant-input sampling interval.
    current : ArrayLike
        One finite real current value per maintained step.
    backend : str, default: "auto"
        ``auto``, ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Complete state/spike trajectories and final receipts.

    Raises
    ------
    ValueError
        If the configuration, current, or backend name is invalid.
    RuntimeError
        If an explicitly requested maintained backend is unavailable.
    FloatingPointError
        If a numerical candidate or backend result violates the contract.
    """
    unit = _unit(x, y, b, omega, threshold, dt)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown resonate-and-fire backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, drive))
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} resonate-and-fire backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, drive))
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(unit.x, unit.y),
        threshold=unit.threshold,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_python",
    "simulate_resonate_and_fire",
]
