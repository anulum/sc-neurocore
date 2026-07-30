# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for the retained SC adaptive-LIF contract."""

from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
Result: TypeAlias = dict[str, FloatArray | npt.NDArray[np.int64] | float]
KERNEL = "sc_non_resetting_adaptive_lif_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2.0e-12, "julia": 2.0e-12, "go": 2.0e-12, "mojo": 2.0e-12}
_AUTO_BACKENDS = with_floor("python")


class _Runner(Protocol):
    """Mapping-returning native batch facade."""

    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_non_resetting_adaptive_lif")


def backend_available(backend: str) -> bool:
    """Return whether one named retained-project runtime is executable now."""
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_sc_non_resetting_adaptive_lif_simulate")
    if backend == "julia":
        try:
            importlib.import_module(
                "sc_neurocore.accel.julia.neurons"
            )._ensure_sc_non_resetting_adaptive_lif_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_SC_NON_RESETTING_ADAPTIVE_LIF", False))
    return False


def auto_backend() -> str:
    """Return the first available measured lane, with Python as floor."""
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(config: tuple[float, ...], currents: FloatArray) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
        SCNonResettingAdaptiveLIFNeuron,
    )

    neuron = SCNonResettingAdaptiveLIFNeuron(
        v=config[0],
        theta=config[1],
        v_rest=config[2],
        theta_rest=config[3],
        delta_theta=config[4],
        tau_m=config[5],
        tau_theta=config[6],
        r_m=config[7],
        dt=config[8],
    )
    voltage = np.empty(currents.size)
    theta = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = neuron.step(float(current))
        voltage[index] = neuron.v
        theta[index] = neuron.theta
    return {
        "voltages": voltage,
        "theta": theta,
        "events": events,
        "v_final": neuron.v,
        "theta_final": neuron.theta,
    }


def _normalise(result: Mapping[str, object], steps: int, initial: tuple[float, ...]) -> Result:
    out: Result = {}
    for key in ("voltages", "theta"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SC adaptive LIF backend returned malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("SC adaptive LIF backend returned malformed events")
    out["events"] = events
    for index, (trace_key, final_key) in enumerate(
        zip(("voltages", "theta"), ("v_final", "theta_final"), strict=True)
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(FloatArray, out[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"SC adaptive LIF {final_key} disagrees with trace")
        out[final_key] = value
    return out


def simulate_sc_non_resetting_adaptive_lif(
    currents: npt.ArrayLike,
    *,
    v: float = -65.0,
    theta: float = -50.0,
    v_rest: float = -65.0,
    theta_rest: float = -50.0,
    delta_theta: float = 5.0,
    tau_m: float = 10.0,
    tau_theta: float = 50.0,
    r_m: float = 1.0,
    dt: float = 0.1,
    backend: str = "auto",
) -> Result:
    """Run the complete retained-project contract on one real backend."""
    config = tuple(
        float(value)
        for value in (v, theta, v_rest, theta_rest, delta_theta, tau_m, tau_theta, r_m, dt)
    )
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("SC adaptive LIF current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SC adaptive LIF backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SC adaptive LIF backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust SC adaptive LIF backend is unavailable")
        result = cast(_Runner, module.py_sc_non_resetting_adaptive_lif_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module(
            "sc_neurocore.accel.julia.neurons"
        ).simulate_sc_non_resetting_adaptive_lif(
            drive,
            **dict(
                zip(
                    (
                        "v",
                        "theta",
                        "v_rest",
                        "theta_rest",
                        "delta_theta",
                        "tau_m",
                        "tau_theta",
                        "r_m",
                        "dt",
                    ),
                    config,
                    strict=True,
                )
            ),
        )
    else:
        result = cast(_Runner, _native_module(selected).simulate_sc_non_resetting_adaptive_lif)(
            *config, drive
        )
    return _normalise(result, drive.size, config[:2])


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_sc_non_resetting_adaptive_lif",
]
