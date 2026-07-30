# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for the complete source MAT(1) contract."""

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
KERNEL = "non_resetting_lif_mat1_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2.0e-12, "julia": 2.0e-12, "go": 2.0e-12, "mojo": 2.0e-12}
_AUTO_BACKENDS = with_floor("python")


class _Runner(Protocol):
    """Mapping-returning contract shared by native batch facades."""

    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.non_resetting_lif")


def backend_available(backend: str) -> bool:
    """Return whether one named MAT(1) runtime is executable now."""
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_non_resetting_lif_simulate")
    if backend == "julia":
        try:
            importlib.import_module(
                "sc_neurocore.accel.julia.neurons"
            )._ensure_non_resetting_lif_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_NON_RESETTING_LIF", False))
    return False


def auto_backend() -> str:
    """Return the first available measured lane, with Python as floor."""
    return next(
        (
            backend
            for backend in select_backend_order(KERNEL, static=_AUTO_BACKENDS)
            if backend_available(backend)
        ),
        "python",
    )


def _python(config: tuple[float, ...], currents: FloatArray) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron

    neuron = NonResettingLIFNeuron(
        v=config[0],
        theta=config[1],
        refractory_remaining=config[2],
        omega=config[3],
        tau_m=config[4],
        tau_theta=config[5],
        alpha=config[6],
        resistance=config[7],
        refractory_period=config[8],
        dt=config[9],
    )
    voltage = np.empty(currents.size)
    theta = np.empty(currents.size)
    refractory = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = neuron.step(float(current))
        voltage[index] = neuron.v
        theta[index] = neuron.theta
        refractory[index] = neuron.refractory_remaining
    return {
        "voltages": voltage,
        "theta": theta,
        "refractory": refractory,
        "events": events,
        "v_final": neuron.v,
        "theta_final": neuron.theta,
        "refractory_final": neuron.refractory_remaining,
    }


def _normalise(result: Mapping[str, object], steps: int, initial: tuple[float, ...]) -> Result:
    out: Result = {}
    for key in ("voltages", "theta", "refractory"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"MAT(1) backend returned malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("MAT(1) backend returned malformed events")
    out["events"] = events
    for index, (trace_key, final_key) in enumerate(
        zip(
            ("voltages", "theta", "refractory"),
            ("v_final", "theta_final", "refractory_final"),
            strict=True,
        )
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(FloatArray, out[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"MAT(1) {final_key} disagrees with trace")
        out[final_key] = value
    return out


def simulate_non_resetting_lif(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    theta: float = 0.0,
    refractory_remaining: float = 0.0,
    omega: float = 19.0,
    tau_m: float = 5.0,
    tau_theta: float = 50.0,
    alpha: float = 37.0,
    resistance: float = 50.0,
    refractory_period: float = 2.0,
    dt: float = 0.001,
    backend: str = "auto",
) -> Result:
    """Run the complete configured source MAT(1) contract on one backend."""
    config = tuple(
        float(value)
        for value in (
            v,
            theta,
            refractory_remaining,
            omega,
            tau_m,
            tau_theta,
            alpha,
            resistance,
            refractory_period,
            dt,
        )
    )
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("MAT(1) current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown MAT(1) backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} MAT(1) backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust MAT(1) backend is unavailable")
        result = cast(_Runner, module.py_non_resetting_lif_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module(
            "sc_neurocore.accel.julia.neurons"
        ).simulate_non_resetting_lif(
            drive,
            **dict(
                zip(
                    (
                        "v",
                        "theta",
                        "refractory_remaining",
                        "omega",
                        "tau_m",
                        "tau_theta",
                        "alpha",
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
        result = cast(_Runner, _native_module(selected).simulate_non_resetting_lif)(*config, drive)
    return _normalise(result, drive.size, config[:3])


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_non_resetting_lif",
]
