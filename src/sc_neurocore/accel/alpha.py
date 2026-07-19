# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse measured-order accelerator dispatch

"""Dispatch and validate the complete exact-flow dual alpha-synapse batch."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.neurons.models.alpha import AlphaNeuron, AlphaResult

KERNEL = "alpha_exact_flow_batch"
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
_TRACE_KEYS = ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes")
_FINAL_KEYS = ("v_final", "a_exc_final", "i_exc_final", "a_inh_final", "i_inh_final")

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
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


class _BatchRunner(Protocol):
    def __call__(
        self,
        v: float,
        a_exc: float,
        i_exc: float,
        a_inh: float,
        i_inh: float,
        v_rest: float,
        v_threshold: float,
        tau_v: float,
        tau_exc: float,
        tau_inh: float,
        dt: float,
        exc_current: npt.NDArray[np.float64],
        inh_current: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _BatchRunner:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_BatchRunner, engine.py_alpha_simulate)


try:
    _engine_simulate: _BatchRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.alpha")


def _ensure_julia_loaded() -> Any:
    """Load the kernel through the shared Julia neuron registry."""
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return registry._ensure_alpha_loaded()


def backend_available(backend: str) -> bool:
    """Return whether one maintained execution lane is ready.

    Parameters
    ----------
    backend : str
        One of ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the named runtime and its Model42 artefact are available.
    """
    if backend == "python":
        return True
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            _ensure_julia_loaded()
        except (ImportError, FileNotFoundError):
            return False
        except Exception as exc:
            if exc.__class__.__name__ == "JuliaError":
                return False
            raise
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        marker = f"_HAS_{backend.upper()}_ALPHA"
        return bool(getattr(module, marker, False))
    return False


def auto_backend() -> str:
    """Return the first available runtime in measured ascending-latency order.

    Returns
    -------
    str
        An available backend name, with ``python`` as the total fallback.
    """
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _input(
    exc_current: npt.ArrayLike,
    inh_current: npt.ArrayLike | float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    exc_logical = np.asarray(exc_current)
    if exc_logical.ndim != 1:
        raise ValueError(f"exc_current must be one-dimensional: got shape {exc_logical.shape}")
    if exc_logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"exc_current exceeds the signed-32-bit step limit: {exc_logical.size}")
    exc_values = np.ascontiguousarray(exc_logical, dtype=np.float64)
    inh_logical = np.asarray(inh_current)
    inh_values: npt.NDArray[np.float64]
    if inh_logical.ndim == 0:
        inh_values = np.full(exc_values.size, float(inh_logical), dtype=np.float64)
    elif inh_logical.ndim == 1 and inh_logical.size == exc_values.size:
        inh_values = np.ascontiguousarray(inh_logical, dtype=np.float64)
    else:
        raise ValueError(
            f"inh_current must be a scalar or match exc_current length: "
            f"got shape {inh_logical.shape} for {exc_values.size} steps"
        )
    if not np.isfinite(exc_values).all() or not np.isfinite(inh_values).all():
        raise ValueError("current values must contain only finite values")
    return exc_values, inh_values


def _unit(
    v: float,
    a_exc: float,
    i_exc: float,
    a_inh: float,
    i_inh: float,
    v_rest: float,
    v_threshold: float,
    tau_v: float,
    tau_exc: float,
    tau_inh: float,
    dt: float,
) -> AlphaNeuron:
    return AlphaNeuron(
        v=v,
        a_exc=a_exc,
        i_exc=i_exc,
        a_inh=a_inh,
        i_inh=i_inh,
        v_rest=v_rest,
        v_threshold=v_threshold,
        tau_v=tau_v,
        tau_exc=tau_exc,
        tau_inh=tau_inh,
        dt=dt,
    )


def _arguments(
    unit: AlphaNeuron,
    exc_drive: npt.NDArray[np.float64],
    inh_drive: npt.NDArray[np.float64],
) -> _BatchArguments:
    return (
        unit.v,
        unit.a_exc,
        unit.i_exc,
        unit.a_inh,
        unit.i_inh,
        unit.v_rest,
        unit.v_threshold,
        unit.tau_v,
        unit.tau_exc,
        unit.tau_inh,
        unit.dt,
        exc_drive,
        inh_drive,
    )


def normalise_result(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float, float, float, float],
    v_rest: float,
) -> AlphaResult:
    """Validate complete state/spike traces and scalar final receipts.

    Parameters
    ----------
    result : dict[str, object]
        Backend mapping with the five state traces, ``spikes``, the five
        final states, and an integral ``spike_count``.
    n_steps : int
        Required length of every trajectory.
    initial : tuple[float, float, float, float, float]
        Initial ``v, a_exc, i_exc, a_inh, i_inh`` used to validate an empty
        batch.
    v_rest : float
        Membrane-potential reset installed at every spike.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Contiguous finite trajectories and mutually consistent final receipts.

    Raises
    ------
    FloatingPointError
        If any backend field is missing, malformed, non-finite, inconsistent,
        non-binary, or violates the somatic reset contract.
    """
    normalised: AlphaResult = {}
    for key in _TRACE_KEYS:
        try:
            values = np.asarray(result[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"alpha backend returned invalid {key} trace") from exc
        if values.ndim != 1 or values.shape != (n_steps,):
            raise FloatingPointError(f"alpha backend returned malformed {key} trace")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"alpha backend returned non-finite {key} trace")
        if key == "spikes" and not np.isin(values, (0.0, 1.0)).all():
            raise FloatingPointError("alpha backend returned a non-binary spike trace")
        normalised[key] = np.ascontiguousarray(values)

    for index, (key, state_key) in enumerate(
        zip(_FINAL_KEYS, ("v", "a_exc", "i_exc", "a_inh", "i_inh"), strict=True)
    ):
        try:
            final = float(cast(float, result[key]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"alpha backend returned invalid {key}") from exc
        if not np.isfinite(final):
            raise FloatingPointError(f"alpha backend returned non-finite {key}")
        trace = cast(npt.NDArray[np.float64], normalised[state_key])
        expected = initial[index] if n_steps == 0 else float(trace[-1])
        if abs(final - expected) > _RESULT_TOLERANCE:
            raise FloatingPointError(f"alpha {key} disagrees with its trace")
        normalised[key] = final

    try:
        raw_spike_count = result["spike_count"]
    except KeyError as exc:
        raise FloatingPointError("alpha backend returned invalid spike_count") from exc
    if isinstance(raw_spike_count, bool) or not isinstance(raw_spike_count, Integral):
        raise FloatingPointError("alpha backend returned invalid spike_count")
    spike_count = int(raw_spike_count)
    spike_trace = cast(npt.NDArray[np.float64], normalised["spikes"])
    if spike_count < 0 or spike_count != int(np.sum(spike_trace, dtype=np.float64)):
        raise FloatingPointError("alpha spike_count disagrees with its trace")
    normalised["spike_count"] = spike_count

    v_trace = cast(npt.NDArray[np.float64], normalised["v"])
    if n_steps:
        reset_mask = spike_trace == 1.0
        if not np.all(v_trace[reset_mask] == v_rest):
            raise FloatingPointError("alpha spike trace disagrees with the somatic v reset")
    return normalised


def simulate_python(
    v: float,
    a_exc: float,
    i_exc: float,
    a_inh: float,
    i_inh: float,
    v_rest: float,
    v_threshold: float,
    tau_v: float,
    tau_exc: float,
    tau_inh: float,
    dt: float,
    exc_current: npt.ArrayLike,
    inh_current: npt.ArrayLike | float = 0.0,
) -> AlphaResult:
    """Run the complete batch through the Python golden model."""
    unit = _unit(v, a_exc, i_exc, a_inh, i_inh, v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt)
    exc_drive, inh_drive = _input(exc_current, inh_current)
    traces = {
        key: np.empty(exc_drive.size, dtype=np.float64)
        for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes")
    }
    spike_count = 0
    for index, (exc_value, inh_value) in enumerate(zip(exc_drive, inh_drive, strict=True)):
        spike = unit.step(float(exc_value), float(inh_value))
        traces["v"][index] = unit.v
        traces["a_exc"][index] = unit.a_exc
        traces["i_exc"][index] = unit.i_exc
        traces["a_inh"][index] = unit.a_inh
        traces["i_inh"][index] = unit.i_inh
        traces["spikes"][index] = spike
        spike_count += spike
    result: dict[str, object] = {
        **traces,
        "v_final": unit.v,
        "a_exc_final": unit.a_exc,
        "i_exc_final": unit.i_exc,
        "a_inh_final": unit.a_inh,
        "i_inh_final": unit.i_inh,
        "spike_count": spike_count,
    }
    return normalise_result(
        result,
        n_steps=exc_drive.size,
        initial=(v, a_exc, i_exc, a_inh, i_inh),
        v_rest=v_rest,
    )


def _simulate_julia(
    v: float,
    a_exc: float,
    i_exc: float,
    a_inh: float,
    i_inh: float,
    v_rest: float,
    v_threshold: float,
    tau_v: float,
    tau_exc: float,
    tau_inh: float,
    dt: float,
    exc_current: npt.NDArray[np.float64],
    inh_current: npt.NDArray[np.float64],
) -> dict[str, object]:
    registry = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return cast(
        "dict[str, object]",
        registry.simulate_alpha(
            v,
            a_exc,
            i_exc,
            a_inh,
            i_inh,
            v_rest,
            v_threshold,
            tau_v,
            tau_exc,
            tau_inh,
            dt,
            exc_current,
            inh_current,
        ),
    )


def _native_runner(backend: str) -> _BatchRunner:
    if backend == "rust":
        if _engine_simulate is None:
            raise RuntimeError("Rust alpha backend is unavailable")
        return _engine_simulate
    if backend == "julia":
        return _simulate_julia
    module = _native_module(backend)
    return cast(_BatchRunner, module.simulate_alpha)


def simulate_alpha(
    v: float = 0.0,
    a_exc: float = 0.0,
    i_exc: float = 0.0,
    a_inh: float = 0.0,
    i_inh: float = 0.0,
    v_rest: float = 0.0,
    v_threshold: float = 1.0,
    tau_v: float = 20.0,
    tau_exc: float = 5.0,
    tau_inh: float = 10.0,
    dt: float = 1.0,
    exc_current: npt.ArrayLike = (),
    inh_current: npt.ArrayLike | float = 0.0,
    *,
    backend: str = "auto",
) -> AlphaResult:
    """Run one complete exact-flow batch on a selected execution lane.

    Parameters
    ----------
    v, a_exc, i_exc, a_inh, i_inh : float, default: 0.0
        Initial membrane potential and synaptic cascade states.
    v_rest : float, default: 0.0
        Leak reversal potential, also the somatic spike reset.
    v_threshold : float, default: 1.0
        Spike threshold; must exceed ``v_rest``.
    tau_v : float, default: 20.0
        Positive membrane time constant.
    tau_exc, tau_inh : float, default: 5.0 / 10.0
        Positive excitatory and inhibitory alpha time constants.
    dt : float, default: 1.0
        Positive piecewise-constant-input sampling interval.
    exc_current : ArrayLike
        One finite real excitatory drive value per maintained step.
    inh_current : ArrayLike or float, default: 0.0
        Inhibitory drive, scalar or matching the excitatory length.
    backend : str, default: "auto"
        ``auto``, ``python``, ``rust``, ``julia``, ``go``, or ``mojo``.

    Returns
    -------
    dict[str, numpy.ndarray | float | int]
        Complete state/spike trajectories and final receipts.

    Raises
    ------
    ValueError
        If the configuration, currents, or backend name is invalid.
    RuntimeError
        If an explicitly requested maintained backend is unavailable.
    FloatingPointError
        If a numerical candidate or backend result violates the contract.
    """
    unit = _unit(v, a_exc, i_exc, a_inh, i_inh, v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt)
    exc_drive, inh_drive = _input(exc_current, inh_current)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in _AUTO_BACKENDS:
        raise ValueError(f"unknown alpha backend: {selected}")
    if selected == "python":
        return simulate_python(*_arguments(unit, exc_drive, inh_drive))
    if not backend_available(selected):
        raise RuntimeError(f"{selected.title()} alpha backend is unavailable")
    result = _native_runner(selected)(*_arguments(unit, exc_drive, inh_drive))
    return normalise_result(
        result,
        n_steps=exc_drive.size,
        initial=(unit.v, unit.a_exc, unit.i_exc, unit.a_inh, unit.i_inh),
        v_rest=unit.v_rest,
    )


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "normalise_result",
    "simulate_alpha",
    "simulate_python",
]
