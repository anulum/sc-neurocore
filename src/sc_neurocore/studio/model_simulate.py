# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model simulation entrypoints

"""Python and optional Rust batch simulation for Studio model runs.

Every run first resolves its effective inputs through
:mod:`sc_neurocore.studio.model_run_contract`, so an invalid request is rejected
before any model is constructed, and a numerical failure is reported with its
step instead of being replaced by a silent zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sc_neurocore_engine.studio import get_batch_simulate
except ImportError:

    def get_batch_simulate() -> object:
        """Return the optional Rust batch simulator or raise when unavailable."""
        raise ImportError("Studio Rust batch simulator unavailable")


from sc_neurocore.studio.model_introspection import _classify_fields
from sc_neurocore.studio.model_run_contract import (
    DriveTrace,
    ModelRunInputs,
    ModelSimulationFailure,
    bounded_diagnostic,
    resolve_drive_trace,
    resolve_model_run_inputs,
    run_receipt,
)
from sc_neurocore.studio.simulation import MAX_PLOT_POINTS, MAX_STEPS, _spike_stats

_RUST_STATE_EXPORTS: tuple[str, ...] = ("v",)


class RustStudioBackendUnavailable(ImportError):
    """Raised when the Studio Rust batch-simulation path is unavailable."""


class RustStudioBackendError(RuntimeError):
    """Raised when the Studio Rust batch-simulation path fails at runtime."""


def _load_rust_batch_simulate() -> Any:
    """Load the Rust batch-simulation bridge entrypoint.

    Import failure means the backend is unavailable; it must not be conflated
    with runtime failure inside an otherwise available backend.
    """
    try:
        return get_batch_simulate()
    except ImportError as exc:
        raise RustStudioBackendUnavailable("Studio Rust batch simulator unavailable") from exc


def _is_rust_unsupported_model_error(exc: Exception) -> bool:
    """Return whether the Rust backend rejected a model as unsupported."""
    return isinstance(exc, ValueError) and "Unsupported model:" in str(exc)


def _plot_stride(n_steps: int) -> int:
    """Return the decimation stride that keeps a trace within ``MAX_PLOT_POINTS``."""
    return n_steps // MAX_PLOT_POINTS if n_steps > MAX_PLOT_POINTS else 1


def _first_non_finite(values: np.ndarray[Any, Any]) -> int | None:
    """Return the index of the first non-finite sample, or ``None`` when all are finite."""
    finite = np.isfinite(values)
    if bool(np.all(finite)):
        return None
    return int(np.argmin(finite))


def _try_rust_simulate(
    name: str,
    n_steps: int,
    current_trace: Any,
    actual_dt: float,
) -> dict[str, Any] | None:
    """Attempt Rust batch simulation.

    Returns ``None`` only when the backend is unavailable or the model is not
    implemented in Rust. Runtime failures in an available backend are raised so
    the caller does not silently degrade to Python; a non-finite voltage trace
    is a :class:`ModelSimulationFailure` at its first non-finite step.
    """
    try:
        py_batch_simulate = _load_rust_batch_simulate()
    except RustStudioBackendUnavailable:
        return None

    current_arr = np.asarray(current_trace, dtype=np.float64)
    try:
        result = py_batch_simulate(name, n_steps, current_arr)
    except Exception as exc:
        if _is_rust_unsupported_model_error(exc):
            return None
        raise RustStudioBackendError(
            f"Studio Rust batch simulation failed for model '{name}'"
        ) from exc

    voltages = np.asarray(result["voltages"], dtype=np.float64)
    bad_step = _first_non_finite(voltages)
    if bad_step is not None:
        raise ModelSimulationFailure(
            model=name,
            backend="rust",
            step=bad_step,
            time_ms=bad_step * actual_dt,
            diagnostic=f"state 'v' became non-finite ({voltages[bad_step]!r})",
        )
    spikes = result["spikes"].tolist()
    stats = _spike_stats(spikes, actual_dt, n_steps)

    stride = _plot_stride(n_steps)
    time = np.arange(n_steps) * actual_dt
    if stride > 1:
        time = time[::stride]
        voltages = voltages[::stride]
        current_arr = current_arr[::stride]

    return {
        "time": time.tolist(),
        "states": {"v": voltages.tolist()},
        "current_trace": current_arr.tolist(),
        "spikes": spikes,
        "spike_count": len(spikes),
        "stats": stats,
        "dt": actual_dt,
        "n_steps": n_steps,
        "model_name": name,
    }


def _scalar_state(value: object) -> float | None:
    """Return ``value`` as a float when it is a real scalar, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


def _state_recording_plan(
    neuron: Any, state_names: list[str]
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Split the catalogue state variables into recordable scalars and declared exclusions."""
    recorded: list[str] = []
    excluded: list[tuple[str, str]] = []
    for state_name in state_names:
        if not hasattr(neuron, state_name):
            excluded.append((state_name, "absent on the model instance"))
            continue
        value = getattr(neuron, state_name)
        if _scalar_state(value) is None:
            excluded.append((state_name, f"non-scalar state ({type(value).__name__})"))
            continue
        recorded.append(state_name)
    return tuple(recorded), tuple(excluded)


def _simulate_python(inputs: ModelRunInputs, trace: DriveTrace) -> dict[str, Any]:
    """Run the Python reference model step by step under the resolved contract."""
    neuron = inputs.instantiate()
    n_steps = trace.n_steps
    dt = inputs.dt
    state_vars, _ = _classify_fields(inputs.cls)
    recorded, excluded = _state_recording_plan(neuron, [s["name"] for s in state_vars])
    traces: dict[str, np.ndarray[Any, Any]] = {
        state_name: np.empty(n_steps) for state_name in recorded
    }
    spike_indices: list[int] = []
    drive = inputs.drive

    for t in range(n_steps):
        sample = trace.samples[t]
        value: float | int = int(sample) if drive.kind == "int" else float(sample)
        try:
            spike = (
                neuron.step(value)
                if drive.positional_only
                else neuron.step(**{drive.parameter: value})
            )
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise ModelSimulationFailure(
                model=inputs.model,
                backend="python",
                step=t,
                time_ms=t * dt,
                diagnostic=bounded_diagnostic(exc),
            ) from exc
        for state_name in recorded:
            raw = getattr(neuron, state_name)
            scalar = _scalar_state(raw)
            if scalar is None:
                raise ModelSimulationFailure(
                    model=inputs.model,
                    backend="python",
                    step=t,
                    time_ms=t * dt,
                    diagnostic=f"state {state_name!r} is no longer a scalar ({type(raw).__name__})",
                )
            if not math.isfinite(scalar):
                raise ModelSimulationFailure(
                    model=inputs.model,
                    backend="python",
                    step=t,
                    time_ms=t * dt,
                    diagnostic=f"state {state_name!r} became non-finite ({scalar!r})",
                )
            traces[state_name][t] = scalar
        if spike:
            spike_indices.append(t)

    stats = _spike_stats(spike_indices, dt, n_steps)
    stride = _plot_stride(n_steps)
    time = np.arange(n_steps) * dt
    samples = trace.samples
    if stride > 1:
        time = time[::stride]
        traces = {state_name: arr[::stride] for state_name, arr in traces.items()}
        samples = samples[::stride]

    return {
        "time": time.tolist(),
        "states": {state_name: arr.tolist() for state_name, arr in traces.items()},
        "current_trace": samples.tolist(),
        "spikes": spike_indices,
        "spike_count": len(spike_indices),
        "stats": stats,
        "dt": dt,
        "n_steps": n_steps,
        "model_name": inputs.model,
        "effective_inputs": run_receipt(
            inputs,
            trace,
            backend="python",
            recorded_state=recorded,
            excluded_state=excluded,
            plot_stride=stride,
        ),
    }


def simulate_model(
    name: str,
    param_overrides: dict[str, float] | None = None,
    dt: float | None = None,
    duration: float = 100.0,
    current: float = 10.0,
    protocol: str = "constant",
    frequency_hz: float = 10.0,
    use_fast_path: bool = True,
) -> dict[str, Any]:
    """Simulate a named catalogue model under a fail-closed input contract.

    Parameters
    ----------
    name : str
        Registered catalogue class name.
    param_overrides : dict[str, float] or None
        Constructor overrides; every key must be an overridable numeric field of
        the model and every value a finite number of the declared kind.
    dt : float or None
        Timestep in milliseconds. ``None`` uses the model default. A model whose
        step is a fixed class attribute accepts only that value; a model without
        any timestep accepts only the Studio default of 0.1 ms.
    duration : float
        Requested run length in milliseconds; capped at ``MAX_STEPS`` steps and
        reported as ``steps_truncated`` in the receipt.
    current : float
        Protocol amplitude; must be finite. Integer-drive models additionally
        require every sample of the protocol to be integral.
    protocol : {"constant", "step", "ramp", "pulse", "sine"}
        Current-injection protocol.
    frequency_hz : float
        Sine frequency; must be positive and finite.
    use_fast_path : bool
        Allow the Rust batch backend when no override or explicit ``dt`` is
        given. The behaviour probe passes ``False`` so its characterisation is
        the canonical Python model's, independent of the loaded extension.

    Returns
    -------
    dict[str, Any]
        Time base, recorded state traces, injected current, spike indices,
        statistics and an ``effective_inputs`` receipt naming the backend, the
        effective parameters, the drive contract and every excluded state.

    Raises
    ------
    ModelInputError
        When any input is unknown, mistyped, non-finite, fractional for an
        integer field, unsupported for the model, or when the model cannot be
        constructed or driven under the request.
    ModelSimulationFailure
        When a step raises or produces a non-finite or non-scalar state; the
        failure names the backend, step index and simulated time.
    RustStudioBackendError
        When an available Rust backend fails for a reason other than an
        unsupported model.
    """
    inputs = resolve_model_run_inputs(name, param_overrides, dt)
    trace = resolve_drive_trace(
        inputs,
        protocol=protocol,
        current=current,
        duration=duration,
        frequency_hz=frequency_hz,
        max_steps=MAX_STEPS,
    )

    if use_fast_path and not inputs.overrides_applied and dt is None:
        rust_result = _try_rust_simulate(name, trace.n_steps, trace.samples, inputs.dt)
        if rust_result is not None:
            state_names = [s["name"] for s in _classify_fields(inputs.cls)[0]]
            excluded = tuple(
                (state_name, "not exported by the Rust batch backend")
                for state_name in state_names
                if state_name not in _RUST_STATE_EXPORTS
            )
            rust_result["effective_inputs"] = run_receipt(
                inputs,
                trace,
                backend="rust",
                recorded_state=_RUST_STATE_EXPORTS,
                excluded_state=excluded,
                plot_stride=_plot_stride(trace.n_steps),
            )
            return rust_result

    return _simulate_python(inputs, trace)
