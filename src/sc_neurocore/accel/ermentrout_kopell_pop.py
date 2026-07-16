# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Montbrió population measured-order accelerator dispatch

"""Dispatch and validate the complete two-state MPR Euler batch."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
    ErmentroutKopellPopulationResult,
)

KERNEL = "ermentrout_kopell_pop_euler_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "mojo": 1.0e-10,
}
_AUTO_BACKENDS = with_floor("python")
_MAX_NATIVE_STEPS = (1 << 31) - 1
_RESULT_TOLERANCE = 1.0e-10
_STATE_KEYS = ("r", "v")
_FINAL_KEYS = ("r_final", "v_final")

_BatchArguments = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    npt.NDArray[np.float64],
]


class _BatchRunner(Protocol):
    """Typed contract shared by each mapping-returning native facade."""

    def __call__(
        self,
        r_init: float,
        v_init: float,
        tau: float,
        delta: float,
        eta_bar: float,
        coupling: float,
        dt: float,
        ext_input: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_ermentrout_kopell_pop_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.ermentrout_kopell_pop")


def backend_available(backend: str) -> bool:
    """Return whether one named execution lane is ready.

    Parameters
    ----------
    backend : str
        One of ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the corresponding runtime and artefact are available.
    """
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            module = importlib.import_module("sc_neurocore.accel.julia.neurons")
            module._ensure_ermentrout_kopell_pop_loaded()
        except (ImportError, FileNotFoundError):
            return False
        except Exception as exc:
            if module.is_julia_error(exc):
                return False
            raise
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        marker = f"_HAS_{backend.upper()}_ERMENTROUT_KOPELL_POP"
        return bool(getattr(module, marker, False))
    return False


def auto_backend() -> str:
    """Return the first available runtime in measured ascending-latency order.

    Returns
    -------
    str
        Available backend name, with ``python`` as the total fallback.
    """
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _input(ext_input: npt.ArrayLike) -> npt.NDArray[np.float64]:
    logical = np.asarray(ext_input)
    if logical.ndim != 1:
        raise ValueError(f"ext_input must be one-dimensional: got shape {logical.shape}")
    if logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"ext_input exceeds the signed-32-bit step limit: {logical.size}")
    values = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ext_input must contain only finite values")
    return values


def _unit(
    r: float,
    v: float,
    tau: float,
    delta: float,
    eta_bar: float,
    coupling: float,
    dt: float,
) -> ErmentroutKopellPopulation:
    return ErmentroutKopellPopulation(
        r=r,
        v=v,
        tau=tau,
        delta=delta,
        eta_bar=eta_bar,
        j=coupling,
        dt=dt,
    )


def _arguments(
    unit: ErmentroutKopellPopulation,
    ext_input: npt.NDArray[np.float64],
) -> _BatchArguments:
    return (
        unit.r,
        unit.v,
        unit.tau,
        unit.delta,
        unit.eta_bar,
        unit.j,
        unit.dt,
        ext_input,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float],
) -> ErmentroutKopellPopulationResult:
    """Validate traces and final-state receipts from one backend.

    Parameters
    ----------
    result : dict[str, object]
        Backend mapping containing ``r``, ``v``, and both final receipts.
    n_steps : int
        Required trace length.
    initial : tuple[float, float]
        Initial ``r`` and ``v`` used to validate an empty batch.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Contiguous finite traces and consistent scalar final receipts.

    Raises
    ------
    FloatingPointError
        If a trace or final receipt is absent, malformed, non-finite,
        physically invalid, or inconsistent with the complete trajectory.
    """
    normalised: ErmentroutKopellPopulationResult = {}
    for key in _STATE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"MPR backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"MPR backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"MPR backend returned non-finite {key} trace")
        if key == "r" and np.any(values < 0.0):
            raise FloatingPointError("MPR backend returned a negative firing-rate trace")
        normalised[key] = np.ascontiguousarray(values)

    for index, (key, state_key) in enumerate(zip(_FINAL_KEYS, _STATE_KEYS, strict=True)):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"MPR backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"MPR backend returned non-finite {key}")
        if key == "r_final" and final < 0.0:
            raise FloatingPointError("MPR backend returned a negative final firing rate")
        trace = cast(npt.NDArray[np.float64], normalised[state_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"MPR {key} disagrees with its trace")
        normalised[key] = final
    return normalised


def simulate_python(
    r: float,
    v: float,
    tau: float,
    delta: float,
    eta_bar: float,
    coupling: float,
    dt: float,
    ext_input: npt.ArrayLike,
) -> ErmentroutKopellPopulationResult:
    """Run the complete batch through the Python golden model.

    Parameters
    ----------
    r, v : float
        Initial population firing rate and mean membrane potential.
    tau, delta, eta_bar, coupling, dt : float
        Complete MPR configuration and explicit-Euler step.
    ext_input : ArrayLike
        One finite external drive per step.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Complete post-update state traces and final-state receipts.

    Raises
    ------
    ValueError
        If the configuration or input vector violates the public contract.
    FloatingPointError
        If a candidate state violates the finite non-negative-rate contract.
    """
    unit = _unit(r, v, tau, delta, eta_bar, coupling, dt)
    drive = _input(ext_input)
    r_trace = np.empty(drive.size, dtype=np.float64)
    v_trace = np.empty(drive.size, dtype=np.float64)
    for index, value in enumerate(drive):
        unit.step(float(value))
        r_trace[index] = unit.r
        v_trace[index] = unit.v
    result: dict[str, object] = {
        "r": r_trace,
        "v": v_trace,
        "r_final": unit.r,
        "v_final": unit.v,
    }
    return normalise_result(result, n_steps=drive.size, initial=(r, v))


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust MPR backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        module = importlib.import_module("sc_neurocore.accel.julia.neurons")
        return cast(_BatchRunner, module.simulate_ermentrout_kopell_pop)
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_ermentrout_kopell_pop)


def simulate_ermentrout_kopell_pop(
    r: float = 0.1,
    v: float = -2.0,
    tau: float = 1.0,
    delta: float = 1.0,
    eta_bar: float = -5.0,
    coupling: float = 15.0,
    dt: float = 0.01,
    ext_input: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> ErmentroutKopellPopulationResult:
    """Run one complete MPR Euler batch on a selected execution lane.

    Parameters
    ----------
    r, v : float
        Initial population firing rate and mean membrane potential.
    tau, delta, eta_bar, coupling, dt : float
        Complete MPR configuration and explicit-Euler step.
    ext_input : ArrayLike
        One finite external drive value per step.
    backend : str, default="auto"
        ``python``, ``rust``, ``julia``, ``go``, ``mojo``, or measured
        ascending-latency selection.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Complete post-update ``r`` and ``v`` traces plus final receipts.

    Raises
    ------
    ValueError
        If the configuration, input vector, or backend name is invalid.
    RuntimeError
        If an explicitly requested compiled backend is unavailable.
    FloatingPointError
        If a backend returns malformed, non-finite, negative-rate, or
        internally inconsistent results.

    Notes
    -----
    All native results pass through :func:`normalise_result`; no partially
    validated backend mapping is returned to the caller.
    """
    unit = _unit(r, v, tau, delta, eta_bar, coupling, dt)
    drive = _input(ext_input)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown MPR backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, drive))
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} MPR backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, drive))
    return normalise_result(result, n_steps=drive.size, initial=(unit.r, unit.v))


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_ermentrout_kopell_pop",
    "simulate_python",
]
