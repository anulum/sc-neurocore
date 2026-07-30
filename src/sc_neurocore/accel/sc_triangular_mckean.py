# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for the retained SC triangular recurrence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.sc_triangular_mckean import SCTriangularMcKeanNeuron

KERNEL = "sc_triangular_mckean_rk4_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 2e-12,
    "julia": 2e-12,
    "go": 2e-12,
    "mojo": 2e-12,
}
_AUTO_BACKENDS = with_floor("python")


def backend_available(backend: str) -> bool:
    """Return whether the retained recurrence can execute on ``backend``."""
    if backend == "python":
        return True
    probe = SCTriangularMcKeanNeuron()
    try:
        probe.simulate(0, backend=backend)
    except (ImportError, RuntimeError):
        return False
    return backend in PARITY_ATOL


def auto_backend() -> str:
    """Select the first available backend under the configured policy."""
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _run(
    config: tuple[float, ...], currents: npt.NDArray[np.float64], backend: str
) -> Mapping[str, object]:
    neuron = SCTriangularMcKeanNeuron(*config)
    voltages = np.empty(currents.size)
    recovery = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        if backend == "python":
            events[index] = neuron.step(float(current))
        else:
            _, spikes = neuron.simulate(1, float(current), backend=backend)
            events[index] = spikes
        voltages[index] = neuron.v
        recovery[index] = neuron.w
    return {
        "voltages": voltages,
        "recovery": recovery,
        "events": events,
        "v_final": neuron.v,
        "w_final": neuron.w,
    }


def simulate_sc_triangular_mckean(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    w: float = 0.0,
    a: float = 0.25,
    epsilon: float = 0.01,
    gamma: float = 0.5,
    dt: float = 0.1,
    v_peak: float = 0.8,
    backend: str = "auto",
) -> dict[str, object]:
    """Execute the complete retained state/event trace on one runtime."""
    config = tuple(float(x) for x in (v, w, a, epsilon, gamma, dt, v_peak))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("SC triangular McKean current must be finite and one-dimensional")
    _run(config, drive[:0], "python")
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SC triangular McKean backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SC triangular McKean backend is unavailable")
    result = _run(config, drive, selected)
    out: dict[str, object] = {}
    for key in ("voltages", "recovery"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (drive.size,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SC triangular McKean backend returned malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (drive.size,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("SC triangular McKean backend returned malformed events")
    out["events"] = events
    for trace_key, final_key, initial in (
        ("voltages", "v_final", config[0]),
        ("recovery", "w_final", config[1]),
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(npt.NDArray[np.float64], out[trace_key])
        expected = initial if drive.size == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"SC triangular McKean {final_key} disagrees with trace")
        out[final_key] = value
    return out


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_sc_triangular_mckean",
]
