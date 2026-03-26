# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ODE simulation engine for Studio Equation Playground

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.equation_builder import from_equations

MAX_STEPS = 100_000
MAX_PLOT_POINTS = 5_000


def simulate(
    equations: list[str],
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    duration: float = 100.0,
    current: float = 0.0,
) -> dict:
    """Run an ODE neuron simulation and return time series data.

    Returns dict with keys: time, states (dict of variable arrays),
    spikes (list of spike time indices), spike_count.
    """
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

    for t in range(n_steps):
        spike = neuron.step(I=current)
        for v in var_names:
            traces[v][t] = neuron.state[v]
        if spike:
            spike_indices.append(t)

    time = np.arange(n_steps) * dt

    # Downsample for browser if trace is large
    if n_steps > MAX_PLOT_POINTS:
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}

    return {
        "time": time.tolist(),
        "states": {v: arr.tolist() for v, arr in traces.items()},
        "spikes": spike_indices,
        "spike_count": len(spike_indices),
        "dt": dt,
        "n_steps": n_steps,
    }
