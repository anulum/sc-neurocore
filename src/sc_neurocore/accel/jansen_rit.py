# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen–Rit measured-order accelerator dispatch

"""Dispatch and validate the complete Jansen–Rit equation-(6) batch."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.jansen_rit import JansenRitResult, JansenRitUnit

KERNEL = "jansen_rit_euler_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-11,
    "julia": 1.0e-11,
    "go": 1.0e-11,
    "mojo": 1.0e-8,
}
_AUTO_BACKENDS = with_floor("python")
_RESULT_TOLERANCE = 1.0e-8
_STATE_KEYS = ("y0", "y3", "y1", "y4", "y2", "y5")
_TRACE_KEYS = (*_STATE_KEYS, "eeg")
_FINAL_KEYS = tuple(f"{key}_final" for key in _STATE_KEYS)

_BatchArguments = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
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
        y0_init: float,
        y3_init: float,
        y1_init: float,
        y4_init: float,
        y2_init: float,
        y5_init: float,
        a_exc: float,
        b_exc: float,
        a_rate: float,
        b_rate: float,
        c: float,
        e0: float,
        v0: float,
        r: float,
        dt: float,
        p_ext: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_jansen_rit_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.jansen_rit")


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
            module._ensure_jansen_rit_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_JANSEN_RIT", False))
    return False


def auto_backend() -> str:
    """Return the first available runtime in measured ascending-latency order."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _input(p_ext: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.ascontiguousarray(p_ext, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"p_ext must be one-dimensional: got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("p_ext must contain only finite values")
    return values


def _unit(
    y0: float,
    y3: float,
    y1: float,
    y4: float,
    y2: float,
    y5: float,
    a_exc: float,
    b_exc: float,
    a_rate: float,
    b_rate: float,
    c: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
) -> JansenRitUnit:
    return JansenRitUnit(
        y0=y0,
        y3=y3,
        y1=y1,
        y4=y4,
        y2=y2,
        y5=y5,
        a_exc=a_exc,
        b_exc=b_exc,
        a_rate=a_rate,
        b_rate=b_rate,
        c=c,
        e0=e0,
        v0=v0,
        r=r,
        dt=dt,
    )


def _arguments(unit: JansenRitUnit, p_ext: npt.NDArray[np.float64]) -> _BatchArguments:
    return (
        unit.y0,
        unit.y3,
        unit.y1,
        unit.y4,
        unit.y2,
        unit.y5,
        unit.a_exc,
        unit.b_exc,
        unit.a_rate,
        unit.b_rate,
        unit.c,
        unit.e0,
        unit.v0,
        unit.r,
        unit.dt,
        p_ext,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float, float, float, float, float],
) -> JansenRitResult:
    """Validate all traces and final-state receipts from one backend."""
    normalised: JansenRitResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Jansen–Rit backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"Jansen–Rit backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Jansen–Rit backend returned non-finite {key} trace")
        normalised[key] = np.ascontiguousarray(values)

    y1_trace = cast(npt.NDArray[np.float64], normalised["y1"])
    y2_trace = cast(npt.NDArray[np.float64], normalised["y2"])
    eeg_trace = cast(npt.NDArray[np.float64], normalised["eeg"])
    if not np.allclose(eeg_trace, y1_trace - y2_trace, rtol=0.0, atol=_RESULT_TOLERANCE):
        raise FloatingPointError("Jansen–Rit EEG trace disagrees with y1 - y2")

    for index, (key, state_key) in enumerate(zip(_FINAL_KEYS, _STATE_KEYS, strict=True)):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Jansen–Rit backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"Jansen–Rit backend returned non-finite {key}")
        trace = cast(npt.NDArray[np.float64], normalised[state_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"Jansen–Rit {key} disagrees with its trace")
        normalised[key] = final
    return normalised


def simulate_python(
    y0: float,
    y3: float,
    y1: float,
    y4: float,
    y2: float,
    y5: float,
    a_exc: float,
    b_exc: float,
    a_rate: float,
    b_rate: float,
    c: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    p_ext: npt.ArrayLike,
) -> JansenRitResult:
    """Run the equation-(6) batch through the Python golden model."""
    unit = _unit(
        y0,
        y3,
        y1,
        y4,
        y2,
        y5,
        a_exc,
        b_exc,
        a_rate,
        b_rate,
        c,
        e0,
        v0,
        r,
        dt,
    )
    drive = _input(p_ext)
    traces = {key: np.empty(drive.size, dtype=np.float64) for key in _TRACE_KEYS}
    for index, value in enumerate(drive):
        eeg = unit.step(float(value))
        for key, state in zip(
            _STATE_KEYS,
            (unit.y0, unit.y3, unit.y1, unit.y4, unit.y2, unit.y5),
            strict=True,
        ):
            traces[key][index] = state
        traces["eeg"][index] = eeg
    result: dict[str, object] = {
        **traces,
        **{f"{key}_final": getattr(unit, key) for key in _STATE_KEYS},
    }
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(y0, y3, y1, y4, y2, y5),
    )


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust Jansen–Rit backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        module = importlib.import_module("sc_neurocore.accel.julia.neurons")
        return cast(_BatchRunner, module.simulate_jansen_rit)
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_jansen_rit)


def simulate_jansen_rit(
    y0: float = 0.0,
    y3: float = 0.0,
    y1: float = 0.0,
    y4: float = 0.0,
    y2: float = 0.0,
    y5: float = 0.0,
    a_exc: float = 3.25,
    b_exc: float = 22.0,
    a_rate: float = 100.0,
    b_rate: float = 50.0,
    c: float = 135.0,
    e0: float = 2.5,
    v0: float = 6.0,
    r: float = 0.56,
    dt: float = 0.0001,
    p_ext: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> JansenRitResult:
    """Run one complete Jansen–Rit batch on a selected execution lane."""
    unit = _unit(
        y0,
        y3,
        y1,
        y4,
        y2,
        y5,
        a_exc,
        b_exc,
        a_rate,
        b_rate,
        c,
        e0,
        v0,
        r,
        dt,
    )
    drive = _input(p_ext)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown Jansen–Rit backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, drive))
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} Jansen–Rit backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, drive))
    return normalise_result(
        result,
        n_steps=drive.size,
        initial=(unit.y0, unit.y3, unit.y1, unit.y4, unit.y2, unit.y5),
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_jansen_rit",
    "simulate_python",
]
