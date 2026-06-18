# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ODE simulation engine for Studio Equation Playground

from __future__ import annotations

from typing import Any
import numpy as np

from sc_neurocore.neurons.equation_builder import from_equations

MAX_STEPS = 100_000
MAX_PLOT_POINTS = 5_000


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
) -> dict[str, Any]:
    """Run an ODE neuron simulation and return time series data."""
    n_steps = int(duration / dt)
    if n_steps > MAX_STEPS:
        n_steps = MAX_STEPS
    if n_steps < 1:
        raise ValueError(f"Duration {duration} with dt {dt} yields < 1 step")

    neuron = from_equations(
        *equations,
        threshold=threshold,
        reset=reset if reset else None,
        params=params,
        init=init,
        dt=dt,
    )

    var_names = list(neuron.state.keys())
    traces = {v: np.empty(n_steps) for v in var_names}
    spike_indices: list[int] = []

    I_trace = _make_current_trace(protocol, current, n_steps, dt=dt, frequency_hz=frequency_hz)

    for t in range(n_steps):
        spike = neuron.step(I=float(I_trace[t]))
        for v in var_names:
            traces[v][t] = neuron.state[v]
        if spike:
            spike_indices.append(t)

    time = np.arange(n_steps) * dt
    stats = _spike_stats(spike_indices, dt, n_steps)

    # Current trace for plotting (downsample with states)
    current_trace = I_trace

    if n_steps > MAX_PLOT_POINTS:
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}  # type: ignore[misc]
        current_trace = current_trace[::stride]

    return {
        "time": time.tolist(),
        "states": {v: arr.tolist() for v, arr in traces.items()},
        "current_trace": current_trace.tolist(),
        "spikes": spike_indices,
        "spike_count": len(spike_indices),
        "stats": stats,
        "dt": dt,
        "n_steps": n_steps,
    }


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
