# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang measured-order accelerator dispatch

"""Dispatch and validate the complete Wong-Wang Euler/OU batch contract."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.wong_wang import WongWangUnit

WongWangResult = dict[str, npt.NDArray[np.float64] | float]
KERNEL = "wong_wang_euler_ou_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "mojo": 1.0e-9,
}
_AUTO_BACKENDS = with_floor("python")
_RESULT_TOLERANCE = 1.0e-9
_TRACE_KEYS = ("s1", "s2", "noise1", "noise2", "r1", "r2")
_FINAL_KEYS = ("s1_final", "s2_final", "noise1_final", "noise2_final")
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
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


class _BatchRunner(Protocol):
    """Typed contract shared by every mapping-returning native facade."""

    def __call__(
        self,
        s1_init: float,
        s2_init: float,
        noise1_init: float,
        noise2_init: float,
        tau_s: float,
        tau_ampa: float,
        gamma: float,
        j_n: float,
        j_cross: float,
        i_0: float,
        sigma: float,
        dt: float,
        stim1: npt.NDArray[np.float64],
        stim2: npt.NDArray[np.float64],
        xi: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_wong_wang_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.wong_wang")


def backend_available(backend: str) -> bool:
    """Return whether one named execution lane is ready.

    Parameters
    ----------
    backend : str
        One of ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the corresponding runtime and compiled artefact exist.
    """
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            module = importlib.import_module("sc_neurocore.accel.julia.neurons")
            module._ensure_wong_wang_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_WONG_WANG", False))
    return False


def auto_backend() -> str:
    """Return the first available backend in ascending measured-latency order.

    Returns
    -------
    str
        Available runtime name, with ``python`` as the fail-safe floor.
    """
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _inputs(
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    stim1_values = np.ascontiguousarray(stim1, dtype=np.float64)
    stim2_values = np.ascontiguousarray(stim2, dtype=np.float64)
    xi_values = np.ascontiguousarray(xi, dtype=np.float64)
    arrays = (stim1_values, stim2_values, xi_values)
    for name, array in zip(("stim1", "stim2", "xi"), arrays):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional: got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    steps = arrays[0].size
    if arrays[1].size != steps:
        raise ValueError(f"stim1 and stim2 length mismatch: {steps} vs {arrays[1].size}")
    if arrays[2].size != 2 * steps:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * steps}): got {arrays[2].size}")
    return stim1_values, stim2_values, xi_values


def _unit(
    s1: float,
    s2: float,
    noise1: float,
    noise2: float,
    tau_s: float,
    tau_ampa: float,
    gamma: float,
    j_n: float,
    j_cross: float,
    i_0: float,
    sigma: float,
    dt: float,
) -> WongWangUnit:
    return WongWangUnit(
        s1=s1,
        s2=s2,
        noise1=noise1,
        noise2=noise2,
        tau_s=tau_s,
        tau_ampa=tau_ampa,
        gamma=gamma,
        j_n=j_n,
        j_cross=j_cross,
        i_0=i_0,
        sigma=sigma,
        dt=dt,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float, float, float],
) -> WongWangResult:
    """Validate a complete backend result before exposing it publicly.

    Parameters
    ----------
    result : dict[str, object]
        Mapping produced by one runtime facade.
    n_steps : int
        Required trace length.
    initial : tuple[float, float, float, float]
        Initial dynamic states used for empty-batch final validation.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Contiguous, finite traces and mutually consistent final states.

    Raises
    ------
    FloatingPointError
        If any trace, range, or final-state invariant is violated.
    """
    normalised: WongWangResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Wong-Wang backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"Wong-Wang backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Wong-Wang backend returned non-finite {key} trace")
        if (
            key in {"s1", "s2"}
            and not np.logical_and(
                values >= -_RESULT_TOLERANCE, values <= 1.0 + _RESULT_TOLERANCE
            ).all()
        ):
            raise FloatingPointError(f"Wong-Wang backend returned out-of-range {key}")
        if key in {"r1", "r2"} and (values < -_RESULT_TOLERANCE).any():
            raise FloatingPointError(f"Wong-Wang backend returned negative {key}")
        normalised[key] = np.ascontiguousarray(values)

    trace_final_indices = ("s1", "s2", "noise1", "noise2")
    for index, (key, trace_key) in enumerate(zip(_FINAL_KEYS, trace_final_indices)):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"Wong-Wang backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"Wong-Wang backend returned non-finite {key}")
        trace = cast(npt.NDArray[np.float64], normalised[trace_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"Wong-Wang {key} disagrees with its trace")
        normalised[key] = final
    return normalised


def _arguments(
    unit: WongWangUnit,
    inputs: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> _BatchArguments:
    return (
        unit.s1,
        unit.s2,
        unit.noise1,
        unit.noise2,
        unit.tau_s,
        unit.tau_ampa,
        unit.gamma,
        unit.j_n,
        unit.j_cross,
        unit.i_0,
        unit.sigma,
        unit.dt,
        inputs[0],
        inputs[1],
        inputs[2],
    )


def simulate_python(
    s1: float,
    s2: float,
    noise1: float,
    noise2: float,
    tau_s: float,
    tau_ampa: float,
    gamma: float,
    j_n: float,
    j_cross: float,
    i_0: float,
    sigma: float,
    dt: float,
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> WongWangResult:
    """Run the deterministic-sample batch through the Python golden model.

    Parameters
    ----------
    s1, s2 : float
        Initial NMDA gating fractions.
    noise1, noise2 : float
        Initial Ornstein-Uhlenbeck input-current states.
    tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt : float
        Published reduced-model parameters.
    stim1, stim2 : ArrayLike
        Per-step external currents.
    xi : ArrayLike
        Interleaved standard-normal samples of length ``2 * n_steps``.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Validated state/rate traces and final dynamic states.

    Raises
    ------
    ValueError
        If a state, parameter, or input violates the numerical contract.
    FloatingPointError
        If a complete candidate result violates a state invariant.
    """
    unit = _unit(s1, s2, noise1, noise2, tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt)
    stim1_values, stim2_values, xi_values = _inputs(stim1, stim2, xi)
    traces = {key: np.empty(stim1_values.size, dtype=np.float64) for key in _TRACE_KEYS}
    for step in range(stim1_values.size):
        rate1, rate2 = unit.step_with_gaussian_samples(
            stim1_values[step], stim2_values[step], xi_values[2 * step], xi_values[2 * step + 1]
        )
        for key, value in (
            ("s1", unit.s1),
            ("s2", unit.s2),
            ("noise1", unit.noise1),
            ("noise2", unit.noise2),
            ("r1", rate1),
            ("r2", rate2),
        ):
            traces[key][step] = value
    result: dict[str, object] = {
        **traces,
        "s1_final": unit.s1,
        "s2_final": unit.s2,
        "noise1_final": unit.noise1,
        "noise2_final": unit.noise2,
    }
    return normalise_result(
        result,
        n_steps=stim1_values.size,
        initial=(s1, s2, noise1, noise2),
    )


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust Wong-Wang backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        module = importlib.import_module("sc_neurocore.accel.julia.neurons")
        return cast(_BatchRunner, module.simulate_wong_wang)
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_wong_wang)


def simulate_wong_wang(
    s1: float = 0.1,
    s2: float = 0.1,
    noise1: float = 0.0,
    noise2: float = 0.0,
    tau_s: float = 0.1,
    tau_ampa: float = 0.002,
    gamma: float = 0.641,
    j_n: float = 0.2609,
    j_cross: float = 0.0497,
    i_0: float = 0.3255,
    sigma: float = 0.02,
    dt: float = 0.0001,
    stim1: npt.ArrayLike = (),
    stim2: npt.ArrayLike = (),
    xi: npt.ArrayLike = (),
    *,
    backend: str = "auto",
) -> WongWangResult:
    """Run one complete Wong-Wang batch on a selected execution lane.

    Parameters
    ----------
    s1, s2, noise1, noise2 : float
        Initial dynamic states.
    tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt : float
        Published reduced-model parameters.
    stim1, stim2 : ArrayLike
        Per-step external currents.
    xi : ArrayLike
        Interleaved standard-normal samples.
    backend : str, default="auto"
        Explicit runtime name or ascending measured-latency selection.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Validated state/rate traces and final dynamic states.

    Raises
    ------
    ValueError
        If inputs or the backend name violate the public contract.
    RuntimeError
        If an explicitly selected runtime is unavailable.
    FloatingPointError
        If a runtime returns malformed or physically invalid output.
    """
    unit = _unit(s1, s2, noise1, noise2, tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt)
    inputs = _inputs(stim1, stim2, xi)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown Wong-Wang backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, inputs))
    if not backend_available(selected):
        raise RuntimeError(f"{selected} Wong-Wang backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, inputs))
    return normalise_result(
        result,
        n_steps=inputs[0].size,
        initial=(unit.s1, unit.s2, unit.noise1, unit.noise2),
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "WongWangResult",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_python",
    "simulate_wong_wang",
]
