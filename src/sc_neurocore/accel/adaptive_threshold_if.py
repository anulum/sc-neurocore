# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold measured-order accelerator dispatch

"""Dispatch and validate the complete exact-relaxation adaptive-threshold batch."""

from __future__ import annotations

import importlib
import math
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.adaptive_threshold_if import (
    AdaptiveThresholdIFNeuron,
    AdaptiveThresholdIFResult,
)

KERNEL = "adaptive_threshold_if_exact_relaxation_batch"
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
_TRACE_KEYS = ("v", "theta", "spikes")
_FINAL_KEYS = ("v_final", "theta_final")

_BatchArguments = tuple[
    float,
    float,
    float,
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
        v: float,
        theta: float,
        v_rest: float,
        v_reset: float,
        theta_rest: float,
        delta_theta: float,
        tau_m: float,
        tau_theta: float,
        dt: float,
        current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_adaptive_threshold_if_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.adaptive_threshold_if")


def _ensure_julia_loaded() -> Any:
    """Load the kernel through the shared Julia neuron registry."""
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return registry._ensure_adaptive_threshold_if_loaded()


def backend_available(backend: str) -> bool:
    """Return whether one maintained execution lane is ready.

    Parameters
    ----------
    backend : str
        One of ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the named runtime and its Model41 artefact are available.
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
        marker = f"_HAS_{backend.upper()}_ADAPTIVE_THRESHOLD_IF"
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
    v: float,
    theta: float,
    v_rest: float,
    v_reset: float,
    theta_rest: float,
    delta_theta: float,
    tau_m: float,
    tau_theta: float,
    dt: float,
) -> AdaptiveThresholdIFNeuron:
    return AdaptiveThresholdIFNeuron(
        v=v,
        theta=theta,
        v_rest=v_rest,
        v_reset=v_reset,
        theta_rest=theta_rest,
        delta_theta=delta_theta,
        tau_m=tau_m,
        tau_theta=tau_theta,
        dt=dt,
    )


def _arguments(
    unit: AdaptiveThresholdIFNeuron,
    current: npt.NDArray[np.float64],
) -> _BatchArguments:
    return (
        unit.v,
        unit.theta,
        unit.v_rest,
        unit.v_reset,
        unit.theta_rest,
        unit.delta_theta,
        unit.tau_m,
        unit.tau_theta,
        unit.dt,
        current,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float],
    v_reset: float,
    theta_rest: float,
    delta_theta: float,
    tau_theta: float,
    dt: float,
) -> AdaptiveThresholdIFResult:
    """Validate complete state/spike traces and scalar final receipts.

    Parameters
    ----------
    result : dict[str, object]
        Backend mapping with ``v``, ``theta``, ``spikes``, both final states,
        and an integral ``spike_count``.
    n_steps : int
        Required length of every trajectory.
    initial : tuple[float, float]
        Initial ``v`` and ``theta`` used to validate an empty batch.
    v_reset : float
        Membrane potential reset installed at every spike.
    theta_rest : float
        Baseline threshold of the exact relaxation.
    delta_theta : float
        Fixed post-spike threshold shift.
    tau_theta : float
        Threshold relaxation time constant.
    dt : float
        Sampling interval.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Contiguous finite trajectories and mutually consistent final receipts.

    Raises
    ------
    FloatingPointError
        If any backend field is missing, malformed, non-finite, inconsistent,
        non-binary, or violates the reset/shift contract.
    """
    normalised: AdaptiveThresholdIFResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(
                f"adaptive-threshold backend returned invalid {key} trace"
            ) from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"adaptive-threshold backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"adaptive-threshold backend returned non-finite {key} trace")
        if key == "spikes" and not np.isin(values, (0.0, 1.0)).all():
            raise FloatingPointError("adaptive-threshold backend returned a non-binary spike trace")
        normalised[key] = np.ascontiguousarray(values)

    for index, (key, state_key) in enumerate(zip(_FINAL_KEYS, ("v", "theta"), strict=True)):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"adaptive-threshold backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"adaptive-threshold backend returned non-finite {key}")
        trace = cast(npt.NDArray[np.float64], normalised[state_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"adaptive-threshold {key} disagrees with its trace")
        normalised[key] = final

    try:
        raw_spike_count = result["spike_count"]
    except KeyError as exc:
        raise FloatingPointError("adaptive-threshold backend returned invalid spike_count") from exc
    if isinstance(raw_spike_count, bool) or not isinstance(raw_spike_count, Integral):
        raise FloatingPointError("adaptive-threshold backend returned invalid spike_count")
    spike_count = int(raw_spike_count)
    spike_trace = cast(npt.NDArray[np.float64], normalised["spikes"])
    if spike_count < 0 or spike_count != int(np.sum(spike_trace, dtype=np.float64)):
        raise FloatingPointError("adaptive-threshold spike_count disagrees with its trace")
    normalised["spike_count"] = spike_count

    v_trace = cast(npt.NDArray[np.float64], normalised["v"])
    theta_trace = cast(npt.NDArray[np.float64], normalised["theta"])
    if n_steps:
        reset_mask = spike_trace == 1.0
        if not np.all(v_trace[reset_mask] == v_reset):
            raise FloatingPointError("adaptive-threshold spike trace disagrees with v reset")
        decay_theta = math.exp(-dt / tau_theta)
        previous_theta = np.concatenate(([initial[1]], theta_trace[:-1]))
        relaxed_theta = theta_rest + (previous_theta - theta_rest) * decay_theta
        expected_theta = relaxed_theta + delta_theta
        if not np.all(
            np.abs(theta_trace[reset_mask] - expected_theta[reset_mask]) <= _RESULT_TOLERANCE
        ):
            raise FloatingPointError(
                "adaptive-threshold spike trace disagrees with the fixed threshold shift"
            )
    return normalised


def simulate_python(
    v: float,
    theta: float,
    v_rest: float,
    v_reset: float,
    theta_rest: float,
    delta_theta: float,
    tau_m: float,
    tau_theta: float,
    dt: float,
    current: npt.ArrayLike,
) -> AdaptiveThresholdIFResult:
    """Run the complete batch through the Python golden model.

    Parameters
    ----------
    v, theta : float
        Initial membrane potential and adaptive threshold.
    v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt : float
        Complete exact-relaxation configuration.
    current : ArrayLike
        One finite real piecewise-constant current per maintained step.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Complete post-update traces, final states, and spike count.

    Raises
    ------
    ValueError
        If the configuration or current vector violates the public contract.
    FloatingPointError
        If a candidate or returned receipt is non-finite.
    """
    unit = _unit(v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt)
    drive = _input(current)
    v_trace = np.empty(drive.size, dtype=np.float64)
    theta_trace = np.empty(drive.size, dtype=np.float64)
    spikes = np.empty(drive.size, dtype=np.float64)
    spike_count = 0
    for index, value in enumerate(drive):
        spike = unit.step(float(value))
        v_trace[index] = unit.v
        theta_trace[index] = unit.theta
        spikes[index] = spike
        spike_count += spike
    result: dict[str, object] = {
        "v": v_trace,
        "theta": theta_trace,
        "spikes": spikes,
        "v_final": unit.v,
        "theta_final": unit.theta,
        "spike_count": spike_count,
    }
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(v, theta),
        v_reset=v_reset,
        theta_rest=theta_rest,
        delta_theta=delta_theta,
        tau_theta=tau_theta,
        dt=dt,
    )


def _simulate_julia(
    v: float,
    theta: float,
    v_rest: float,
    v_reset: float,
    theta_rest: float,
    delta_theta: float,
    tau_m: float,
    tau_theta: float,
    dt: float,
    current: npt.NDArray[np.float64],
) -> dict[str, object]:
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return cast(
        "dict[str, object]",
        registry.simulate_adaptive_threshold_if(
            v,
            theta,
            v_rest,
            v_reset,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            dt,
            current,
        ),
    )


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust adaptive-threshold backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return _simulate_julia
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_adaptive_threshold_if)


def simulate_adaptive_threshold_if(
    v: float = -65.0,
    theta: float = -50.0,
    v_rest: float = -65.0,
    v_reset: float = -65.0,
    theta_rest: float = -50.0,
    delta_theta: float = 5.0,
    tau_m: float = 10.0,
    tau_theta: float = 50.0,
    dt: float = 0.1,
    current: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> AdaptiveThresholdIFResult:
    """Run one complete exact-relaxation batch on a selected execution lane.

    Parameters
    ----------
    v : float, default: -65.0
        Initial membrane potential in millivolts.
    theta : float, default: -50.0
        Initial adaptive threshold in millivolts.
    v_rest : float, default: -65.0
        Leak reversal potential in millivolts.
    v_reset : float, default: -65.0
        Post-spike membrane reset in millivolts.
    theta_rest : float, default: -50.0
        Baseline threshold in millivolts; must exceed ``v_rest`` and ``v_reset``.
    delta_theta : float, default: 5.0
        Fixed non-negative post-spike threshold shift in millivolts.
    tau_m : float, default: 10.0
        Positive membrane time constant in milliseconds.
    tau_theta : float, default: 50.0
        Positive threshold relaxation time constant in milliseconds.
    dt : float, default: 0.1
        Positive piecewise-constant-input sampling interval in milliseconds.
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
    unit = _unit(v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt)
    drive = _input(current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown adaptive-threshold backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, drive))
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} adaptive-threshold backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, drive))
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(unit.v, unit.theta),
        v_reset=unit.v_reset,
        theta_rest=unit.theta_rest,
        delta_theta=unit.delta_theta,
        tau_theta=unit.tau_theta,
        dt=unit.dt,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_adaptive_threshold_if",
    "simulate_python",
]
