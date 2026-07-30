# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for retained bipolar accumulator."""

from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

KERNEL = "sc_sigma_delta_accumulator_batch"
PARITY_ATOL = {"python": 0.0, "rust": 0.0, "julia": 0.0, "go": 0.0, "mojo": 0.0}
_AUTO_BACKENDS = with_floor("python")


class _Runner(Protocol):
    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_sigma_delta_accumulator")


def backend_available(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_sc_sigma_delta_accumulator_simulate")
    if backend == "julia":
        try:
            importlib.import_module(
                "sc_neurocore.accel.julia.neurons"
            )._ensure_sc_sigma_delta_accumulator_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_SC_SIGMA_DELTA_ACCUMULATOR", False))
    return False


def auto_backend() -> str:
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(config: tuple[float, ...], currents: npt.NDArray[np.float64]) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.sc_sigma_delta_accumulator import SCSigmaDeltaAccumulatorNeuron

    n = SCSigmaDeltaAccumulatorNeuron(*config)
    sigma = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for i, current in enumerate(currents):
        events[i] = n.step(float(current))
        sigma[i] = n.sigma
    return {"sigma": sigma, "events": events, "sigma_final": n.sigma}


def simulate_sc_sigma_delta_accumulator(
    currents: npt.ArrayLike, *, sigma: float = 0.0, v_threshold: float = 1.0, backend: str = "auto"
) -> dict[str, object]:
    """Run the complete retained project contract on one real backend."""
    config = (float(sigma), float(v_threshold))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("SC SigmaDelta current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SC SigmaDelta backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SC SigmaDelta backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust SC SigmaDelta backend is unavailable")
        result = cast(_Runner, module.py_sc_sigma_delta_accumulator_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module(
            "sc_neurocore.accel.julia.neurons"
        ).simulate_sc_sigma_delta_accumulator(drive, sigma=config[0], v_threshold=config[1])
    else:
        result = cast(_Runner, _native_module(selected).simulate_sc_sigma_delta_accumulator)(
            *config, drive
        )
    trace = np.ascontiguousarray(result["sigma"], dtype=np.float64)
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    final = float(cast(float, result["sigma_final"]))
    expected = config[0] if drive.size == 0 else float(trace[-1])
    if (
        trace.shape != (drive.size,)
        or not np.isfinite(trace).all()
        or events.shape != (drive.size,)
        or not np.isin(events, (-1, 0, 1)).all()
        or final != expected
    ):
        raise FloatingPointError("SC SigmaDelta backend returned malformed trace")
    return {"sigma": trace, "events": events, "sigma_final": final}


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_sc_sigma_delta_accumulator",
]
