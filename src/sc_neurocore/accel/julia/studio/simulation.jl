# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/simulation

module SimulationAccel

using Statistics, LinearAlgebra

function simulate(equations, threshold, reset, params, init, dt, duration, current, protocol)
    equations: list[str],
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    init: dict[str, float] | nothing = nothing,
    dt: float = 0.1,
    duration: float = 100.0,
    current: float = 0.0,
    protocol: str = "constant",
    ) -> dict
    n_steps = int(duration / dt)
    if n_steps > MAX_STEPS
        n_steps = MAX_STEPS
    if n_steps < 1
        raise ValueError(f"Duration {duration} with dt {dt} yields < 1 step")
    neuron = from_equations(
        *equations,
        threshold=threshold,
        reset=reset if reset else nothing,
        params=params,
        init=init,
        dt=dt,
    )
    var_names = list(neuron.state.keys())
    traces = {v: np.empty(n_steps) for v in var_names}
    spike_indices: list[int] = []
    I_trace = _make_current_trace(protocol, current, n_steps)
    for t in 1:n_steps
        spike = neuron.step(I=float(I_trace[t]))
        for v in var_names
            traces[v][t] = neuron.state[v]
        if spike
            spike_indices = push!(, t)
    time = collect(n_steps) * dt
    stats = _spike_stats(spike_indices, dt, n_steps)
    # Current trace for plotting (downsample with states)
    current_trace = I_trace
    if n_steps > MAX_PLOT_POINTS
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}  # type: ignore[misc]
        current_trace = current_trace[::stride]
    return {
        "time": time.tolist(),
        "states": {v: arr.tolist() for v, arr in traces.items()},
        "current_trace": current_trace.tolist(),
        "spikes": spike_indices,
        "spike_count": length(spike_indices),
        "stats": stats,
        "dt": dt,
        "n_steps": n_steps,
    }
end

function fi_curve(equations, threshold, reset, params, init, dt, duration, i_min, i_max, i_steps)
    equations: list[str],
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    init: dict[str, float] | nothing = nothing,
    dt: float = 0.1,
    duration: float = 200.0,
    i_min: float = 0.0,
    i_max: float = 50.0,
    i_steps: int = 20,
    ) -> dict
    currents = range(i_min, i_max, i_steps).tolist()
    rates: list[float] = []
    for I_val in currents
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
        rates = push!(, result["stats"]["rate_hz"])
    return {"currents": currents, "rates": rates}
end

end # module SimulationAccel
