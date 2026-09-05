# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ODE simulation engine for Studio Equation Playground

"""Equation-playground simulation with complete raw custody.

The state layout of an equation neuron is the set of variables the equations
declare; the run records every one of them at every step under the post-step
observation clock and returns the raw arrays next to a bounded display
projection (:mod:`sc_neurocore.studio.trace_projection`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.studio.state_layout import (
    StateObservationError,
    equation_state,
    observe_layout,
    read_variable,
    snapshot,
)
from sc_neurocore.studio.trace_projection import (
    MAX_PLOT_POINTS,
    RAW_ELEMENT_BUDGET,
    custody_payload,
)

MAX_STEPS = 100_000
ODE_MODEL_NAME = "ode"


def _spike_stats(spike_indices: list[int], dt: float, n_steps: int) -> dict[str, Any]:
    """Compute spike statistics from spike index list."""
    duration_s = n_steps * dt / 1000.0
    rate = len(spike_indices) / duration_s if duration_s > 0 else 0.0
    if len(spike_indices) < 2:
        return {
            "rate_hz": round(rate, 2),
            "isi_mean_ms": None,
            "isi_cv": None,
            "isi_histogram": None,
        }
    isis = np.diff(spike_indices).astype(float) * dt
    isi_mean = float(np.mean(isis))
    isi_std = float(np.std(isis))
    isi_cv = isi_std / isi_mean if isi_mean > 0 else 0.0
    # ISI histogram (10 bins)
    counts, edges = np.histogram(isis, bins=min(15, max(3, len(isis) // 3)))
    return {
        "rate_hz": round(rate, 2),
        "isi_mean_ms": round(isi_mean, 3),
        "isi_cv": round(isi_cv, 4),
        "isi_histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }


def _make_current_trace(
    protocol: str,
    current: float,
    n_steps: int,
    dt: float = 0.1,
    frequency_hz: float = 10.0,
    step_onset: float = 0.2,
    step_offset: float = 0.8,
    ramp_start: float = 0.0,
    ramp_end: float | None = None,
) -> np.ndarray[Any, Any]:
    """Generate a current injection trace for the given protocol."""
    I = np.zeros(n_steps)
    if protocol == "constant":
        I[:] = current
    elif protocol == "step":
        i0 = int(n_steps * step_onset)
        i1 = int(n_steps * step_offset)
        I[i0:i1] = current
    elif protocol == "ramp":
        end = ramp_end if ramp_end is not None else current
        I[:] = np.linspace(ramp_start, end, n_steps)
    elif protocol == "pulse":
        period = max(n_steps // 5, 10)
        on_dur = max(period // 5, 2)
        for start in range(0, n_steps, period):
            I[start : start + on_dur] = current
    elif protocol == "sine":
        t_ms = np.arange(n_steps) * dt
        I[:] = current * np.sin(2 * np.pi * frequency_hz * t_ms / 1000.0)
    else:
        I[:] = current
    return I


def _run_failure(step: int, dt: float, exc: BaseException) -> Exception:
    """Build the structured numerical-failure error for an equation run.

    The error class lives in :mod:`sc_neurocore.studio.model_run_contract`,
    which imports the protocol builder above; the import is deferred to the
    failure path to keep the module graph acyclic.
    """
    from sc_neurocore.studio.model_run_contract import ModelSimulationFailure, bounded_diagnostic

    diagnostic = (
        f"state {exc.name!r} {exc.reason}"
        if isinstance(exc, StateObservationError)
        else bounded_diagnostic(exc)
    )
    return ModelSimulationFailure(
        model=ODE_MODEL_NAME,
        backend="python",
        step=step,
        time_ms=step * dt,
        diagnostic=diagnostic,
    )


def simulate(
    equations: list[str],
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    duration: float = 100.0,
    current: float = 0.0,
    protocol: str = "constant",
    frequency_hz: float = 10.0,
    seed: int | None = None,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Run an equation-neuron simulation and return its complete raw result.

    Parameters
    ----------
    equations : list[str]
        Differential equations or map updates, one per state variable.
    threshold, reset : str or None
        Spike condition and reset assignments.
    params, init : dict or None
        Parameter values and initial state.
    dt, duration : float
        Step in milliseconds and requested run length; capped at
        ``max_steps`` steps (:data:`MAX_STEPS` by default; the experiment
        contract refuses a longer synchronous run instead of shortening it).
    current, protocol, frequency_hz : float, str, float
        Injection protocol.
    seed : int or None
        Seed of the diffusion-noise generator (the ``xi`` symbol). ``None``
        draws from the process-global ``numpy.random`` stream, which is not
        reproducible; the experiment contract always passes a seed for a
        stochastic playground run and rejects one for noise-free equations.

    Returns
    -------
    dict[str, Any]
        The display projection (``time``, ``states``, ``current_trace``),
        spikes and statistics, the ``observation`` clock, the ``state_layout``
        (source ``equations``), exact ``initial_state`` and ``final_state``
        snapshots, the full-resolution ``raw`` block and the ``display``
        sample-index map.

    Raises
    ------
    ValueError
        When the duration yields no complete step or the equations are invalid.
    ModelSimulationFailure
        When a step raises or a state variable becomes non-finite (reported
        with the step index and the time that step started, never as a NaN
        trace).
    """
    n_steps = int(duration / dt)
    if n_steps > max_steps:
        n_steps = max_steps
    if n_steps < 1:
        raise ValueError(f"Duration {duration} with dt {dt} yields < 1 step")

    neuron = from_equations(
        *equations,
        threshold=threshold,
        reset=reset if reset else None,
        params=dict(params) if params else None,
        init=dict(init) if init else None,
        dt=dt,
        noise_rng=np.random.default_rng(seed) if seed is not None else None,
    )

    var_names = list(neuron.state.keys())
    layout = observe_layout(
        neuron.state,
        "equations",
        "",
        equation_state(var_names, init),
        n_steps=n_steps,
        element_budget=RAW_ELEMENT_BUDGET,
    )
    try:
        initial_state = snapshot(neuron.state, layout)
    except StateObservationError as exc:
        raise _run_failure(0, dt, exc) from exc
    traces = {v.name: np.empty(n_steps, dtype=np.float64) for v in layout.scalars}
    spike_indices: list[int] = []

    I_trace = _make_current_trace(protocol, current, n_steps, dt=dt, frequency_hz=frequency_hz)

    for t in range(n_steps):
        try:
            spike = neuron.step(I=float(I_trace[t]))
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise _run_failure(t, dt, exc) from exc
        for variable in layout.scalars:
            try:
                traces[variable.name][t] = read_variable(neuron.state, variable)
            except StateObservationError as exc:
                raise _run_failure(t, dt, exc) from exc
        if spike:
            spike_indices.append(t)

    try:
        final_state = snapshot(neuron.state, layout)
    except StateObservationError as exc:
        raise _run_failure(n_steps - 1, dt, exc) from exc
    stats = _spike_stats(spike_indices, dt, n_steps)
    return custody_payload(
        dt=dt,
        n_steps=n_steps,
        layout=layout,
        initial_state=initial_state,
        final_state=final_state,
        scalar_traces=traces,
        vector_traces={},
        vector_omitted=(),
        drive=I_trace,
        spikes=spike_indices,
        stats=stats,
        max_points=MAX_PLOT_POINTS,
    )


def fi_curve(
    equations: list[str],
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    duration: float = 200.0,
    i_min: float = 0.0,
    i_max: float = 50.0,
    i_steps: int = 20,
) -> dict[str, Any]:
    """Sweep current and compute firing rate at each level."""
    currents = np.linspace(i_min, i_max, i_steps).tolist()
    rates: list[float] = []
    for I_val in currents:
        result = simulate(
            equations=equations,
            threshold=threshold,
            reset=reset,
            params=params,
            init=init,
            dt=dt,
            duration=duration,
            current=I_val,
            protocol="constant",
        )
        rates.append(result["stats"]["rate_hz"])
    return {"currents": currents, "rates": rates}


__all__ = [
    "MAX_PLOT_POINTS",
    "MAX_STEPS",
    "ODE_MODEL_NAME",
    "fi_curve",
    "simulate",
]
