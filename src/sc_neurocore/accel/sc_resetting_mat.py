# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — explicit five-runtime SC resetting-MAT dispatch

"""Dispatch the complete project candidate-first RK4/reset trace contract."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
SCResettingMATResult: TypeAlias = dict[str, FloatArray | npt.NDArray[np.int64] | float]
KERNEL = "sc_resetting_mat_rk4_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2.0e-12, "julia": 2.0e-12, "go": 2.0e-12, "mojo": 2.0e-12}
_AUTO_BACKENDS = with_floor("python")


class _NativeRunner(Protocol):
    """Mapping-returning contract shared by native batch facades."""

    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_resetting_mat")


def backend_available(backend: str) -> bool:
    """Return whether one named SC resetting-MAT runtime is executable."""
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_sc_resetting_mat_simulate")
    if backend == "julia":
        try:
            importlib.import_module(
                "sc_neurocore.accel.julia.neurons"
            )._ensure_sc_resetting_mat_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_SC_RESETTING_MAT", False))
    return False


def auto_backend() -> str:
    """Return the first available measured lane, with Python as floor."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _normalise(
    result: Mapping[str, object], steps: int, initial: tuple[float, ...]
) -> SCResettingMATResult:
    normalised: SCResettingMATResult = {}
    trace_keys = ("voltages", "theta1", "theta2")
    for key in trace_keys:
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SC resetting-MAT backend returned malformed {key}")
        normalised[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("SC resetting-MAT backend returned malformed events")
    normalised["events"] = events
    for index, (trace_key, final_key) in enumerate(
        zip(trace_keys, ("v_final", "theta1_final", "theta2_final"), strict=True)
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(FloatArray, normalised[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"SC resetting-MAT {final_key} disagrees with its trace")
        normalised[final_key] = value
    return normalised


def _python_runner(config: tuple[float, ...], currents: FloatArray) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

    neuron = SCResettingMATNeuron(
        v=config[0],
        theta1=config[1],
        theta2=config[2],
        v_rest=config[3],
        v_reset=config[4],
        v_threshold_base=config[5],
        tau_m=config[6],
        tau_1=config[7],
        tau_2=config[8],
        h1=config[9],
        h2=config[10],
        resistance=config[11],
        dt=config[12],
    )
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(3)]
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = neuron.step(float(current))
        traces[0][index] = neuron.v
        traces[1][index] = neuron.theta1
        traces[2][index] = neuron.theta2
    return {
        "voltages": traces[0],
        "theta1": traces[1],
        "theta2": traces[2],
        "events": events,
        "v_final": neuron.v,
        "theta1_final": neuron.theta1,
        "theta2_final": neuron.theta2,
    }


def simulate_sc_resetting_mat(
    currents: npt.ArrayLike,
    *,
    v: float = -70.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
    v_rest: float = -70.0,
    v_reset: float = -70.0,
    v_threshold_base: float = -50.0,
    tau_m: float = 10.0,
    tau_1: float = 10.0,
    tau_2: float = 200.0,
    h1: float = 5.0,
    h2: float = 3.0,
    resistance: float = 1.0,
    dt: float = 1.0,
    backend: str = "auto",
) -> SCResettingMATResult:
    """Run the complete configured SC recurrence on one real backend."""
    config = tuple(
        float(value)
        for value in (
            v,
            theta1,
            theta2,
            v_rest,
            v_reset,
            v_threshold_base,
            tau_m,
            tau_1,
            tau_2,
            h1,
            h2,
            resistance,
            dt,
        )
    )
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("SC resetting-MAT current must be finite and one-dimensional")
    _python_runner(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SC resetting-MAT backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SC resetting-MAT backend is unavailable")
    if selected == "python":
        result = _python_runner(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust SC resetting-MAT backend is unavailable")
        result = cast(_NativeRunner, module.py_sc_resetting_mat_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module(
            "sc_neurocore.accel.julia.neurons"
        ).simulate_sc_resetting_mat(
            drive,
            **dict(
                zip(
                    (
                        "v",
                        "theta1",
                        "theta2",
                        "v_rest",
                        "v_reset",
                        "v_threshold_base",
                        "tau_m",
                        "tau_1",
                        "tau_2",
                        "h1",
                        "h2",
                        "resistance",
                        "dt",
                    ),
                    config,
                    strict=True,
                )
            ),
        )
    else:
        result = cast(_NativeRunner, _native_module(selected).simulate_sc_resetting_mat)(
            *config, drive
        )
    return _normalise(result, drive.size, config[:3])


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_sc_resetting_mat",
]
