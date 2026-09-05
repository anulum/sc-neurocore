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
step instead of being replaced by a silent zero. The Python path records the
model-declared state layout (:mod:`sc_neurocore.studio.state_layout`) at every
step and returns the complete raw result next to a separate display projection
(:mod:`sc_neurocore.studio.trace_projection`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sc_neurocore_engine.studio import get_batch_simulate
except ImportError:

    def get_batch_simulate() -> object:
        """Return the optional Rust batch simulator or raise when unavailable."""
        raise ImportError("Studio Rust batch simulator unavailable")


from sc_neurocore.studio.model_run_contract import (
    DriveTrace,
    ModelRunInputs,
    ModelSimulationFailure,
    bounded_diagnostic,
    resolve_drive_trace,
    resolve_model_run_inputs,
    run_receipt,
)
from sc_neurocore.studio.simulation import MAX_STEPS, _spike_stats
from sc_neurocore.studio.state_layout import (
    ObservedState,
    StateLayout,
    StateObservationError,
    attribute_fingerprints,
    declared_state,
    observe_layout,
    read_variable,
    snapshot,
    undeclared_mutations,
)
from sc_neurocore.studio.trace_projection import RAW_ELEMENT_BUDGET, custody_payload

_RUST_STATE_EXPORTS: tuple[str, ...] = ("v",)
_RUST_INITIAL_SNAPSHOT_NOTE = "the Rust batch backend exposes no initial snapshot"


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


def _first_non_finite(values: np.ndarray[Any, Any]) -> int | None:
    """Return the index of the first non-finite sample, or ``None`` when all are finite."""
    finite = np.isfinite(values)
    if bool(np.all(finite)):
        return None
    return int(np.argmin(finite))


def _state_recording(layout: StateLayout) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Split the layout into per-step recorded names and named exclusions."""
    recorded = tuple(variable.name for variable in layout.per_step)
    excluded: list[tuple[str, str]] = []
    for variable in layout.variables:
        if not variable.observable:
            excluded.append((variable.name, variable.reason))
        elif variable.trace == "snapshots-only":
            excluded.append(
                (variable.name, "vector recorded in snapshots only (raw element budget)")
            )
    return recorded, tuple(excluded)


def _rust_layout(name: str) -> StateLayout:
    """Return the declared layout as the Rust batch backend can honour it."""
    source, stem, declared = declared_state(name)
    variables = tuple(
        ObservedState(spec, "scalar", (), True, "", "per-step")
        if spec.name in _RUST_STATE_EXPORTS
        else ObservedState(
            spec, None, None, False, "not exported by the Rust batch backend", "none"
        )
        for spec in declared
    )
    return StateLayout(
        source=source,
        schema_profile=stem,
        variables=variables,
        custody_notes=(_RUST_INITIAL_SNAPSHOT_NOTE,),
    )


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
    is a :class:`ModelSimulationFailure` at its first non-finite step. The
    result carries the declared layout with every variable the backend does
    not export marked unobservable and no initial snapshot, so it is never
    presented as complete-state custody.
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
    spikes = [int(step) for step in np.asarray(result["spikes"]).tolist()]
    stats = _spike_stats(spikes, actual_dt, n_steps)
    layout = _rust_layout(name)
    payload = custody_payload(
        dt=actual_dt,
        n_steps=n_steps,
        layout=layout,
        initial_state={},
        final_state={"v": float(voltages[-1])},
        scalar_traces={"v": voltages},
        vector_traces={},
        vector_omitted=(),
        drive=current_arr,
        spikes=spikes,
        stats=stats,
    )
    payload["initial_state"] = None
    payload["model_name"] = name
    return payload


def _simulate_python(inputs: ModelRunInputs, trace: DriveTrace) -> dict[str, Any]:
    """Run the Python reference model step by step under the resolved contract."""
    neuron = inputs.instantiate()
    n_steps = trace.n_steps
    dt = inputs.dt
    source, stem, declared = declared_state(inputs.model)
    layout = observe_layout(
        neuron, source, stem, declared, n_steps=n_steps, element_budget=RAW_ELEMENT_BUDGET
    )
    try:
        initial_state = snapshot(neuron, layout)
    except StateObservationError as exc:
        raise ModelSimulationFailure(
            model=inputs.model,
            backend="python",
            step=0,
            time_ms=0.0,
            diagnostic=f"initial state {exc.name!r} {exc.reason}",
        ) from exc
    fingerprints_before = attribute_fingerprints(neuron)

    scalar_traces: dict[str, np.ndarray[Any, Any]] = {}
    vector_traces: dict[str, np.ndarray[Any, Any]] = {}
    vector_omitted: list[str] = []
    for variable in layout.observable:
        if variable.kind == "scalar":
            scalar_traces[variable.name] = np.empty(n_steps, dtype=np.float64)
        elif variable.trace == "per-step" and variable.shape is not None:
            vector_traces[variable.name] = np.empty((n_steps, *variable.shape), dtype=np.float64)
        else:
            vector_omitted.append(variable.name)
    per_step = layout.per_step
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
        for variable in per_step:
            try:
                observed = read_variable(neuron, variable)
            except StateObservationError as exc:
                raise ModelSimulationFailure(
                    model=inputs.model,
                    backend="python",
                    step=t,
                    time_ms=t * dt,
                    diagnostic=f"state {exc.name!r} {exc.reason}",
                ) from exc
            if variable.kind == "scalar":
                scalar_traces[variable.name][t] = observed
            else:
                vector_traces[variable.name][t] = observed
        if spike:
            spike_indices.append(t)

    try:
        final_state = snapshot(neuron, layout)
    except StateObservationError as exc:
        raise ModelSimulationFailure(
            model=inputs.model,
            backend="python",
            step=n_steps - 1,
            time_ms=(n_steps - 1) * dt,
            diagnostic=f"final state {exc.name!r} {exc.reason}",
        ) from exc
    layout = layout.with_undeclared_mutable(
        undeclared_mutations(fingerprints_before, attribute_fingerprints(neuron), layout)
    )
    stats = _spike_stats(spike_indices, dt, n_steps)
    payload = custody_payload(
        dt=dt,
        n_steps=n_steps,
        layout=layout,
        initial_state=initial_state,
        final_state=final_state,
        scalar_traces=scalar_traces,
        vector_traces=vector_traces,
        vector_omitted=vector_omitted,
        drive=trace.samples,
        spikes=spike_indices,
        stats=stats,
    )
    payload["model_name"] = inputs.model
    recorded, excluded = _state_recording(layout)
    payload["effective_inputs"] = run_receipt(
        inputs,
        trace,
        backend="python",
        recorded_state=recorded,
        excluded_state=excluded,
        display_points=int(payload["display"]["point_count"]),
    )
    return payload


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
        given. The Rust result exports the membrane voltage only and no
        initial snapshot, and says so in its ``state_layout``; callers that
        need complete-state custody pass ``False``.

    Returns
    -------
    dict[str, Any]
        The display projection (``time``, ``states``, ``current_trace``),
        spike indices and statistics, the ``observation`` clock, the
        ``state_layout`` with its custody verdict, exact ``initial_state`` and
        ``final_state`` snapshots, the full-resolution ``raw`` block, the
        ``display`` sample-index map and an ``effective_inputs`` receipt.

    Raises
    ------
    ModelInputError
        When any input is unknown, mistyped, non-finite, fractional for an
        integer field, unsupported for the model, or when the model cannot be
        constructed or driven under the request.
    ModelSimulationFailure
        When a step raises or a recorded variable is non-finite or changes
        kind or shape; the failure names the backend, the step index and the
        simulated time at which that step started (``step * dt``).
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
            recorded, excluded = _state_recording(_rust_layout(name))
            rust_result["effective_inputs"] = run_receipt(
                inputs,
                trace,
                backend="rust",
                recorded_state=recorded,
                excluded_state=excluded,
                display_points=int(rust_result["display"]["point_count"]),
            )
            return rust_result

    return _simulate_python(inputs, trace)


__all__ = [
    "RustStudioBackendError",
    "RustStudioBackendUnavailable",
    "simulate_model",
]
