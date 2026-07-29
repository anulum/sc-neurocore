# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — measured-order Amari neural-field accelerator dispatch

"""Dispatch the complete Amari vector recurrence without surrogate fallback."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
AmariResult: TypeAlias = dict[str, FloatArray]
KERNEL = "amari_field_euler_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2.0e-12, "julia": 2.0e-12, "go": 2.0e-12, "mojo": 2.0e-10}
_AUTO_BACKENDS = with_floor("python")


class _NativeRunner(Protocol):
    """Mapping-returning contract shared by non-Rust native facades."""

    def __call__(
        self,
        u_init: FloatArray,
        tau: float,
        a_exc: float,
        a_width: float,
        b_inh: float,
        b_width: float,
        dx: float,
        dt: float,
        currents: FloatArray,
    ) -> Mapping[str, object]: ...


def _engine_class() -> type[Any] | None:
    try:
        module = importlib.import_module("sc_neurocore_engine")
        return cast(type[Any], module.AmariNeuralField)
    except (ImportError, AttributeError):
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.amari_field")


def backend_available(backend: str) -> bool:
    """Return whether one named Amari execution lane is currently usable."""
    if backend == "python":
        return True
    if backend == "rust":
        return _engine_class() is not None
    if backend == "julia":
        try:
            module = importlib.import_module("sc_neurocore.accel.julia.neurons")
            module._ensure_amari_field_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_AMARI_FIELD", False))
    return False


def auto_backend() -> str:
    """Return the first available lane in measured latency order."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _inputs(u_init: npt.ArrayLike, currents: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    state = np.ascontiguousarray(u_init, dtype=np.float64)
    if state.ndim != 1 or state.size < 2 or not np.isfinite(state).all():
        raise ValueError("u_init must be a finite vector with at least two sites")
    drive = np.asarray(currents, dtype=np.float64)
    if drive.ndim == 1:
        drive = np.repeat(drive[:, None], state.size, axis=1)
    if drive.ndim != 2 or drive.shape[1] != state.size:
        raise ValueError(f"currents must have shape (steps,) or (steps, {state.size})")
    if not np.isfinite(drive).all():
        raise ValueError("currents must contain only finite values")
    return state, np.ascontiguousarray(drive)


def _normalise(
    result: Mapping[str, object], steps: int, n: int, initial: FloatArray
) -> AmariResult:
    normalized: AmariResult = {}
    for key, shape in (("states", (steps, n)), ("mean_rates", (steps,)), ("final_state", (n,))):
        try:
            values = np.ascontiguousarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Amari backend returned invalid {key}") from exc
        if values.shape != shape or not np.isfinite(values).all():
            raise FloatingPointError(f"Amari backend returned malformed {key}")
        normalized[key] = values
    rates = normalized["mean_rates"]
    if np.any((rates < 0.0) | (rates > 1.0)):
        raise FloatingPointError("Amari mean rates must lie in [0, 1]")
    expected = initial if steps == 0 else normalized["states"][-1]
    if not np.array_equal(normalized["final_state"], expected):
        raise FloatingPointError("Amari final state disagrees with its state trace")
    return normalized


def _simulate_python(
    state: FloatArray,
    tau: float,
    a_exc: float,
    a_width: float,
    b_inh: float,
    b_width: float,
    dx: float,
    dt: float,
    drive: FloatArray,
) -> AmariResult:
    from sc_neurocore.neurons.models.amari_field import AmariNeuralField

    field = AmariNeuralField(
        n=state.size,
        tau=tau,
        a_exc=a_exc,
        a_width=a_width,
        b_inh=b_inh,
        b_width=b_width,
        dx=dx,
        dt=dt,
        u=state,
    )
    states = np.empty_like(drive)
    rates = np.empty(drive.shape[0], dtype=np.float64)
    for index, row in enumerate(drive):
        rates[index] = field.step(row)
        current_state = field.u
        if current_state is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Amari field state was not initialized")
        states[index] = current_state
    final_state = field.u
    if final_state is None:  # pragma: no cover - constructor invariant
        raise RuntimeError("Amari field state was not initialized")
    return {"states": states, "mean_rates": rates, "final_state": final_state.copy()}


def _simulate_rust(
    state: FloatArray,
    tau: float,
    a_exc: float,
    a_width: float,
    b_inh: float,
    b_width: float,
    dx: float,
    dt: float,
    drive: FloatArray,
) -> AmariResult:
    engine_class = _engine_class()
    if engine_class is None:
        raise RuntimeError("Rust Amari field backend is unavailable")
    signature = str(getattr(engine_class, "__text_signature__", ""))
    if "tau" not in signature:
        module = sys.modules.get(engine_class.__module__)
        origin = getattr(module, "__file__", "unknown")
        raise RuntimeError(f"Rust Amari field extension is stale: {signature} from {origin}")
    field = engine_class(
        n=state.size,
        tau=tau,
        a_exc=a_exc,
        a_width=a_width,
        b_inh=b_inh,
        b_width=b_width,
        dx=dx,
        dt=dt,
        u=state.tolist(),
    )
    states = np.empty_like(drive)
    rates = np.empty(drive.shape[0], dtype=np.float64)
    for index, row in enumerate(drive):
        rates[index] = field.step(row.tolist())
        states[index] = np.asarray(field.get_state(), dtype=np.float64)
    final = state.copy() if drive.shape[0] == 0 else states[-1].copy()
    return {"states": states, "mean_rates": rates, "final_state": final}


def simulate_amari_field(
    u_init: npt.ArrayLike = (),
    tau: float = 10.0,
    a_exc: float = 1.5,
    a_width: float = 2.0,
    b_inh: float = 0.75,
    b_width: float = 1.0,
    dx: float = 0.5,
    dt: float = 0.5,
    currents: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> AmariResult:
    """Run a complete Amari field batch on an explicit maintained backend."""
    state, drive = _inputs(u_init, currents)
    # Instantiation is the common fail-closed configuration validator.
    from sc_neurocore.neurons.models.amari_field import AmariNeuralField

    AmariNeuralField(
        n=state.size,
        tau=tau,
        a_exc=a_exc,
        a_width=a_width,
        b_inh=b_inh,
        b_width=b_width,
        dx=dx,
        dt=dt,
        u=state,
    )
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown Amari field backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} Amari field backend is unavailable")
    arguments = (state, tau, a_exc, a_width, b_inh, b_width, dx, dt, drive)
    result: Mapping[str, object]
    if selected == "python":
        result = _simulate_python(*arguments)
    elif selected == "rust":
        result = _simulate_rust(*arguments)
    elif selected == "julia":
        module = importlib.import_module("sc_neurocore.accel.julia.neurons")
        result = cast(Mapping[str, object], module.simulate_amari_field(*arguments))
    else:
        runner = cast(_NativeRunner, _native_module(selected).simulate_amari_field)
        result = runner(*arguments)
    return _normalise(result, drive.shape[0], state.size, state)


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_amari_field"]
