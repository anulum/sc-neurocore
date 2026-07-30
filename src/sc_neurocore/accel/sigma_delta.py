# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Explicit five-runtime dispatch for sampled source APSDM."""

from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

KERNEL = "sigma_delta_apsdm_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2e-12, "julia": 2e-12, "go": 2e-12, "mojo": 2e-12}
_AUTO_BACKENDS = with_floor("python")


class _Runner(Protocol):
    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust_module() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sigma_delta")


def backend_available(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust_module()
        return module is not None and hasattr(module, "py_sigma_delta_simulate")
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_sigma_delta_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_SIGMA_DELTA", False))
    return False


def auto_backend() -> str:
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(config: tuple[float, ...], currents: npt.NDArray[np.float64]) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.sigma_delta import SigmaDeltaNeuron

    n = SigmaDeltaNeuron(*config)
    sigma = np.empty(currents.size)
    reconstruction = np.empty(currents.size)
    events = np.empty(currents.size, dtype=np.int64)
    for i, current in enumerate(currents):
        events[i] = n.step(float(current))
        sigma[i] = n.sigma
        reconstruction[i] = n.reconstruction
    return {
        "sigma": sigma,
        "reconstruction": reconstruction,
        "events": events,
        "sigma_final": n.sigma,
        "reconstruction_final": n.reconstruction,
    }


def _normalise(
    result: Mapping[str, object], steps: int, initial: tuple[float, ...]
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in ("sigma", "reconstruction"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SigmaDelta backend returned malformed {key}")
        out[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("SigmaDelta backend returned malformed events")
    out["events"] = events
    for i, (trace_key, final_key) in enumerate(
        (("sigma", "sigma_final"), ("reconstruction", "reconstruction_final"))
    ):
        value = float(cast(float, result[final_key]))
        trace = cast(npt.NDArray[np.float64], out[trace_key])
        expected = initial[i] if steps == 0 else float(trace[-1])
        if not np.isfinite(value) or value != expected:
            raise FloatingPointError(f"SigmaDelta {final_key} disagrees with trace")
        out[final_key] = value
    return out


def simulate_sigma_delta(
    currents: npt.ArrayLike,
    *,
    sigma: float = 0.0,
    reconstruction: float = 0.0,
    delta: float = 1.0,
    tau_reconstruction: float = 10.0,
    dt: float = 0.1,
    backend: str = "auto",
) -> dict[str, object]:
    """Run the complete sampled source contract on one real backend."""
    config = tuple(float(v) for v in (sigma, reconstruction, delta, tau_reconstruction, dt))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("SigmaDelta current must be finite and one-dimensional")
    _python(config, drive[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SigmaDelta backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SigmaDelta backend is unavailable")
    if selected == "python":
        result = _python(config, drive)
    elif selected == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("rust SigmaDelta backend is unavailable")
        result = cast(_Runner, module.py_sigma_delta_simulate)(*config, drive)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_sigma_delta(
            drive,
            **dict(
                zip(
                    ("sigma", "reconstruction", "delta", "tau_reconstruction", "dt"),
                    config,
                    strict=True,
                )
            ),
        )
    else:
        result = cast(_Runner, _native_module(selected).simulate_sigma_delta)(*config, drive)
    return _normalise(result, drive.size, config[:2])


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_sigma_delta"]
