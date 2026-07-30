# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for the source McKean system."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

KERNEL = "mckean_rk4_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 2e-12,
    "julia": 2e-12,
    "go": 2e-12,
    "mojo": 2e-12,
}
_AUTO_BACKENDS = with_floor("python")
_NAMES = ("v", "w", "a", "lambda_", "mu", "b", "dt")


class _Runner(Protocol):
    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.mckean")


def backend_available(backend: str) -> bool:
    """Return whether an executable implementation of ``backend`` is available."""
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_mckean_simulate")
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_mckean_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_MCKEAN", False))
    return False


def auto_backend() -> str:
    """Select the first available backend under the configured policy."""
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(config: tuple[float, ...], currents: npt.NDArray[np.float64]) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.mckean import McKeanNeuron

    neuron = McKeanNeuron(*config)
    voltages = np.empty(currents.size)
    recovery = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = neuron.step(float(current))
        voltages[index] = neuron.v
        recovery[index] = neuron.w
    return {
        "voltages": voltages,
        "recovery": recovery,
        "events": events,
        "v_final": neuron.v,
        "w_final": neuron.w,
    }


def _normalise(
    result: Mapping[str, object], steps: int, initial: tuple[float, float]
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in ("voltages", "recovery"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"McKean backend returned malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("McKean backend returned malformed events")
    out["events"] = events
    for index, (trace_key, final_key) in enumerate(
        (("voltages", "v_final"), ("recovery", "w_final"))
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(npt.NDArray[np.float64], out[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"McKean {final_key} disagrees with trace")
        out[final_key] = value
    return out


def simulate_mckean(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    w: float = 0.0,
    a: float = 0.25,
    lambda_: float = 1.0,
    mu: float = 1.0,
    b: float = 0.01,
    dt: float = 0.1,
    backend: str = "auto",
) -> dict[str, object]:
    """Execute the complete source state/event trace on one selected runtime."""
    config = tuple(float(x) for x in (v, w, a, lambda_, mu, b, dt))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("McKean current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown McKean backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} McKean backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust McKean backend is unavailable")
        result = cast(_Runner, module.py_mckean_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_mckean(
            drive, **dict(zip(_NAMES, config, strict=True))
        )
    else:
        result = cast(_Runner, _native_module(selected).simulate_mckean)(*config, drive)
    return _normalise(result, drive.size, (config[0], config[1]))


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_mckean"]
