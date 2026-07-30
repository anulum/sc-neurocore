# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — explicit five-runtime source MAT* dispatch

"""Dispatch the complete non-resetting MAT* trace contract."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
MATResult: TypeAlias = dict[str, FloatArray | npt.NDArray[np.int64] | float]
KERNEL = "mat_star_batch"
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
    return importlib.import_module(f"sc_neurocore.accel.{backend}.mat")


def backend_available(backend: str) -> bool:
    """Return whether one named MAT* runtime is executable now."""
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_mat_simulate")
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_mat_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_MAT", False))
    return False


def auto_backend() -> str:
    """Return the first available measured lane, with Python as floor."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _normalise(result: Mapping[str, object], steps: int, initial: tuple[float, ...]) -> MATResult:
    normalised: MATResult = {}
    trace_keys = ("voltages", "theta1", "theta2", "refractory")
    for key in trace_keys:
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"MAT backend returned malformed {key}")
        normalised[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("MAT backend returned malformed events")
    normalised["events"] = events
    final_keys = ("v_final", "theta1_final", "theta2_final", "refractory_final")
    for index, (trace_key, final_key) in enumerate(zip(trace_keys, final_keys, strict=True)):
        value = float(cast(float, result[final_key]))
        trace = cast(FloatArray, normalised[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"MAT {final_key} disagrees with its trace")
        normalised[final_key] = value
    if np.any(cast(FloatArray, normalised["refractory"]) < 0.0):
        raise FloatingPointError("MAT refractory state became negative")
    return normalised


def _python_runner(config: tuple[float, ...], currents: FloatArray) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.mat import MATNeuron

    neuron = MATNeuron(
        v=config[0],
        theta1=config[1],
        theta2=config[2],
        refractory_remaining=config[3],
        omega=config[4],
        tau_m=config[5],
        tau_1=config[6],
        tau_2=config[7],
        alpha_1=config[8],
        alpha_2=config[9],
        resistance=config[10],
        refractory_period=config[11],
        dt=config[12],
    )
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(4)]
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = neuron.step(float(current))
        traces[0][index] = neuron.v
        traces[1][index] = neuron.theta1
        traces[2][index] = neuron.theta2
        traces[3][index] = neuron.refractory_remaining
    return {
        "voltages": traces[0],
        "theta1": traces[1],
        "theta2": traces[2],
        "refractory": traces[3],
        "events": events,
        "v_final": neuron.v,
        "theta1_final": neuron.theta1,
        "theta2_final": neuron.theta2,
        "refractory_final": neuron.refractory_remaining,
    }


def simulate_mat(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
    refractory_remaining: float = 0.0,
    omega: float = 19.0,
    tau_m: float = 5.0,
    tau_1: float = 10.0,
    tau_2: float = 200.0,
    alpha_1: float = 37.0,
    alpha_2: float = 2.0,
    resistance: float = 50.0,
    refractory_period: float = 2.0,
    dt: float = 0.001,
    backend: str = "auto",
) -> MATResult:
    """Run the complete configured MAT* contract on one real backend."""
    config = tuple(
        float(value)
        for value in (
            v,
            theta1,
            theta2,
            refractory_remaining,
            omega,
            tau_m,
            tau_1,
            tau_2,
            alpha_1,
            alpha_2,
            resistance,
            refractory_period,
            dt,
        )
    )
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("MAT current must be a finite one-dimensional array")
    _python_runner(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown MAT backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} MAT backend is unavailable")
    if selected == "python":
        result = _python_runner(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust MAT backend is unavailable")
        result = cast(_NativeRunner, module.py_mat_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_mat(
            drive,
            **dict(
                zip(
                    (
                        "v",
                        "theta1",
                        "theta2",
                        "refractory_remaining",
                        "omega",
                        "tau_m",
                        "tau_1",
                        "tau_2",
                        "alpha_1",
                        "alpha_2",
                        "resistance",
                        "refractory_period",
                        "dt",
                    ),
                    config,
                    strict=True,
                )
            ),
        )
    else:
        result = cast(_NativeRunner, _native_module(selected).simulate_mat)(*config, drive)
    return _normalise(result, drive.size, config[:4])


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_mat"]
