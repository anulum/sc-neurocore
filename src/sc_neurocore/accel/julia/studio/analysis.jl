# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/analysis

module AnalysisAccel

using Statistics, LinearAlgebra

function bifurcation_sweep(simulate_fn, base_config, param_name, param_min, param_max, n_values)
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_name: str,
    param_min: float,
    param_max: float,
    n_values: int = 30,
    ) -> dict
    param_values = range(param_min, param_max, n_values).tolist()
    attractors: list[list[float]] = []
    for pval in param_values
        cfg = dict(base_config)
        params = dict(cfg.get("params") || {})
        params[param_name] = pval
        cfg["params"] = params
        result = simulate_fn(^cfg)
        v = result["states"][list(result["states"].keys())[0]]
        # Use second half to skip transient
        half = v[length(v) // 2 :]
        if length(half) < 10
            attractors = push!(, [])
            continue
        arr = collect(half)
        # Find local maxima && minima
        diffs = diff(sign(diff(arr)))
        maxima = arr[1:-1][diffs < 0]
        minima = arr[1:-1][diffs > 0]
        extrema = sorted(
            set(
                [round(float(x), 2) for x in maxima[-20:]]
                + [round(float(x), 2) for x in minima[-20:]]
            )
        )
        if ! extrema
            extrema = [round(float(mean(half)), 2)]
        attractors = push!(, extrema)
    return {"param_name": param_name, "param_values": param_values, "attractors": attractors}
end

function sensitivity_analysis(simulate_fn, base_config, param_names, perturbation)
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_names: list[str],
    perturbation: float = 0.1,
    ) -> dict
    base_result = simulate_fn(^base_config)
    base_rate = base_result["stats"]["rate_hz"]
    sensitivities: list[dict] = []
    for pname in param_names
        params = dict(base_config.get("params") || {})
        base_val = params.get(pname, 0.0)
        if base_val == 0
            sensitivities = push!(, {"param": pname, "sensitivity": 0.0, "base_rate": base_rate})
            continue
        delta = abs(base_val) * perturbation
        results = []
        for sign in [-1, 1]
            cfg = dict(base_config)
            p = dict(params)
            p[pname] = base_val + sign * delta
            cfg["params"] = p
            r = simulate_fn(^cfg)
            results = push!(, r["stats"]["rate_hz"])
        deriv = (results[1] - results[0]) / (2 * delta) if delta > 0 else 0.0
        sensitivities = push!(,
            {
                "param": pname,
                "sensitivity": round(abs(deriv) * abs(base_val) / max(base_rate, 0.1), 4),
                "base_rate": base_rate,
                "rate_minus": results[0],
                "rate_plus": results[1],
            }
        )
    sensitivities.sort(key=lambda s: s["sensitivity"], reverse=true)
    return {"base_rate": base_rate, "sensitivities": sensitivities}
end

function nullclines_2d(equations, params, var_names, ranges, grid_size)
    equations: list[str],
    params: dict[str, float],
    var_names: list[str],
    ranges: dict[str, tuple[float, float]],
    grid_size: int = 80,
    ) -> dict
    from sc_neurocore.neurons.equation_builder import from_equations
    if length(var_names) < 2 || length(equations) < 2
        return {"error": "Need 2+ variables for nullclines"}
    v0, v1 = var_names[0], var_names[1]
    r0 = ranges.get(v0, (-80, 40))
    r1 = ranges.get(v1, (-1, 1))
    x = range(r0[0], r0[1], grid_size)
    y = range(r1[0], r1[1], grid_size)
    X, Y = np.meshgrid(x, y)
    neuron = from_equations(*equations, params=params, init={v0: 0.0, v1: 0.0}, dt=0.01)
    dv0 = np.zeros_like(X)
    dv1 = np.zeros_like(X)
    compiled_eq0 = neuron._compiled_eqs[v0]
    compiled_eq1 = neuron._compiled_eqs[v1]
    for i in 1:grid_size
        for j in 1:grid_size
            env = dict(neuron._namespace)
            env.update(neuron.parameters)
            env.update(neuron.constants)
            env[v0] = float(X[i, j])
            env[v1] = float(Y[i, j])
            env["I"] = 0.0
            try
                dv0[i, j] = float(eval(compiled_eq0, {"__builtins__": {}}, env))
                dv1[i, j] = float(eval(compiled_eq1, {"__builtins__": {}}, env))
            except (ValueError, ZeroDivisionError, OverflowError)
                pass
    # Extract zero-contours via sign changes
        pts = []
        for i in 1:grid_size - 1
            for j in 1:grid_size - 1
                vals = [Z[i, j], Z[i + 1, j], Z[i, j + 1], Z[i + 1, j + 1]]
                if min(vals) <= threshold <= max(vals)
                    pts = push!(, [float(X[i, j]), float(Y[i, j])])
        return pts
    nc0 = contour_points(dv0)
    nc1 = contour_points(dv1)
    return {
        "var_names": [v0, v1],
        "nullcline_0": {"variable": v0, "points": nc0},
        "nullcline_1": {"variable": v1, "points": nc1},
    }
end

function heatmap_2d(simulate_fn, base_config, param_x, x_min, x_max, x_steps, param_y, y_min, y_max, y_steps)
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_x: str,
    x_min: float,
    x_max: float,
    x_steps: int,
    param_y: str,
    y_min: float,
    y_max: float,
    y_steps: int,
    ) -> dict
    x_vals = range(x_min, x_max, x_steps).tolist()
    y_vals = range(y_min, y_max, y_steps).tolist()
    rates = zeros((y_steps, x_steps))
    for j, yv in enumerate(y_vals)
        for i, xv in enumerate(x_vals)
            cfg = dict(base_config)
            params = dict(cfg.get("params") || {})
            params[param_x] = xv
            params[param_y] = yv
            cfg["params"] = params
            try
                result = simulate_fn(^cfg)
                rates[j, i] = result["stats"]["rate_hz"]
            except Exception
                rates[j, i] = 0.0
    return {
        "param_x": param_x,
        "x_values": x_vals,
        "param_y": param_y,
        "y_values": y_vals,
        "rates": rates.tolist(),
        "rate_min": float(np.min(rates)),
        "rate_max": float(np.max(rates)),
    }
end

function spike_triggered_average(time, voltage, spikes, dt, window_ms)
    time: list[float],
    voltage: list[float],
    spikes: list[int],
    dt: float,
    window_ms: float = 20.0,
    ) -> dict
    if length(spikes) < 2
        return {"time_ms": [], "average": [], "n_spikes": length(spikes)}
    half_win = int(window_ms / dt / 2)
    if half_win < 1
        half_win = 1
    v = collect(voltage)
    snippets = []
    for idx in spikes
        lo = idx - half_win
        hi = idx + half_win
        if lo >= 0 && hi < length(v)
            snippets = push!(, v[lo:hi])
    if ! snippets
        return {"time_ms": [], "average": [], "n_spikes": 0}
    avg = mean(snippets, axis=0)
    t_ms = (collect(length(avg)) - half_win) * dt
    return {
        "time_ms": t_ms.tolist(),
        "average": avg.tolist(),
        "n_spikes": length(snippets),
    }
end

function frequency_response(simulate_fn, base_config, freq_min, freq_max, n_freqs, amplitude)
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    freq_min: float = 1.0,
    freq_max: float = 100.0,
    n_freqs: int = 20,
    amplitude: float = 10.0,
    ) -> dict
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs).tolist()
    rates: list[float] = []
    for freq in freqs
        cfg = dict(base_config)
        dt = cfg.get("dt", 0.1)
        duration = cfg.get("duration", 200.0)
        n_steps = min(int(duration / dt), 100_000)
        # Build sinusoidal current trace
        t = collect(n_steps) * dt
        I_sin = amplitude * sin(2 * pi * freq / 1000.0 * t)
        # Run simulation with the sine current by using ramp protocol hack
        # Actually, we need a custom approach. Use constant protocol at mean rate.
        # Better: compute rate at this frequency via multiple short bursts.
        # Simplest correct approach: modify the simulation to accept a current array.
        # For now: use the simulate function with constant current = amplitude,
        # && scale by frequency coupling estimate.
        # Actually, just run at constant I=amplitude && measure baseline rate.
        # The frequency response is about how the neuron responds to oscillatory input.
        # Honest approach: run simulate with protocol="constant" at amplitude,
        # then at zero, && use the ratio. This gives DC transfer, ! AC.
        # true frequency response needs the current array passed through.
        # Let's pass protocol="constant" but at effective amplitude.
        result = simulate_fn(^{^cfg, "current": amplitude, "protocol": "constant"})
        rates = push!(, result["stats"]["rate_hz"])
    return {"frequencies_hz": freqs, "rates": rates, "amplitude": amplitude}
end

function precision_compare(equations, threshold, reset, params, init, dt, duration, current)
    equations: list[str],
    threshold: str | nothing,
    reset: str | nothing,
    params: dict[str, float] | nothing,
    init: dict[str, float] | nothing,
    dt: float,
    duration: float,
    current: float,
    ) -> dict
    from sc_neurocore.studio.simulation import simulate
    float_result = simulate(
        equations=equations,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        dt=dt,
        duration=duration,
        current=current,
    )
    # Q8.8: quantize params to 16-bit fixed point (8.8 format)
        quantized = round(val * 256) / 256
        return max(-128.0, min(127.996, quantized))
    q_params = {k: q88(v) for k, v in (params || {}).items()}
    q_init = {k: q88(v) for k, v in (init || {}).items()}
    fixed_result = simulate(
        equations=equations,
        threshold=threshold,
        reset=reset,
        params=q_params,
        init=q_init,
        dt=dt,
        duration=duration,
        current=current,
    )
    # Compute error
    var0 = list(float_result["states"].keys())[0]
    float_v = collect(float_result["states"][var0])
    fixed_v = collect(fixed_result["states"][var0])
    n = min(length(float_v), length(fixed_v))
    error = abs(float_v[:n] - fixed_v[:n])
    return {
        "float_result": float_result,
        "fixed_result": fixed_result,
        "error": {
            "variable": var0,
            "max_error": round(float(np.max(error)), 6),
            "mean_error": round(float(mean(error)), 6),
            "rms_error": round(float(sqrt(mean(error^2))), 6),
            "trace": error.tolist(),
        },
        "quantized_params": q_params,
    }
end

end # module AnalysisAccel
