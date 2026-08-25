# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Explicit five-runtime dispatch for source-faithful Benda-Herz dynamics."""

from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

KERNEL = "benda_herz_rk4_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2e-12, "julia": 2e-12, "go": 2e-12, "mojo": 2e-12}
_AUTO_BACKENDS = with_floor("python")
_NAMES = ("a", "phase", "onset_gain", "rheobase", "adaptation_slope", "tau_a", "dt")


class _Runner(Protocol):
    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.benda_herz")


def backend_available(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_benda_herz_simulate")
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_benda_herz_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_BENDA_HERZ", False))
    return False


def auto_backend() -> str:
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(config: tuple[float, ...], currents: npt.NDArray[np.float64]) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron

    n = BendaHerzNeuron(*config)
    adaptation = np.empty(currents.size)
    phases = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for index, current in enumerate(currents):
        events[index] = n.step(float(current))
        adaptation[index] = n.a
        phases[index] = n.phase
    return {
        "adaptation": adaptation,
        "phases": phases,
        "events": events,
        "a_final": n.a,
        "phase_final": n.phase,
    }


def simulate_benda_herz(
    currents: npt.ArrayLike,
    *,
    a: float = 0.0,
    phase: float = 0.0,
    onset_gain: float = 60.0,
    rheobase: float = 0.0,
    adaptation_slope: float = 0.1,
    tau_a: float = 100.0,
    dt: float = 0.1,
    backend: str = "auto",
) -> dict[str, object]:
    config = tuple(float(x) for x in (a, phase, onset_gain, rheobase, adaptation_slope, tau_a, dt))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Benda-Herz current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown Benda-Herz backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} Benda-Herz backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust Benda-Herz backend is unavailable")
        result = cast(_Runner, module.py_benda_herz_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_benda_herz(
            drive, **dict(zip(_NAMES, config, strict=True))
        )
    else:
        result = cast(_Runner, _native_module(selected).simulate_benda_herz)(*config, drive)
    out: dict[str, object] = {}
    for key in ("adaptation", "phases"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (drive.size,) or not np.isfinite(values).all():
            raise FloatingPointError(f"Benda-Herz malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (drive.size,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("Benda-Herz malformed events")
    out["events"] = events
    for trace_key, final_key, initial in (
        ("adaptation", "a_final", config[0]),
        ("phases", "phase_final", config[1]),
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(npt.NDArray[np.float64], out[trace_key])
        expected = initial if not drive.size else float(trace[-1])
        if value != expected:
            raise FloatingPointError(f"Benda-Herz {final_key} disagrees with trace")
        out[final_key] = value
    return out


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_benda_herz"]
