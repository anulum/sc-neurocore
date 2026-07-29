# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research analysis functions for Studio

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bifurcation_sweep(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_name: str,
    param_min: float,
    param_max: float,
    n_values: int = 30,
) -> dict[str, Any]:
    """Sweep one parameter and extract voltage attractors at each value.

    Returns {param_values, attractors} where attractors[i] is a list
    of voltage extrema in the second half of the simulation (the attractor).
    """
    param_values = np.linspace(param_min, param_max, n_values).tolist()
    attractors: list[list[float]] = []

    for pval in param_values:
        cfg = dict(base_config)
        params = dict(cfg.get("params") or {})
        params[param_name] = pval
        cfg["params"] = params

        result = simulate_fn(**cfg)
        v = result["states"][list(result["states"].keys())[0]]
        # Use second half to skip transient
        half = v[len(v) // 2 :]
        if len(half) < 10:
            attractors.append([])
            continue
        arr = np.array(half)
        # Find local maxima and minima
        diffs = np.diff(np.sign(np.diff(arr)))
        maxima = arr[1:-1][diffs < 0]
        minima = arr[1:-1][diffs > 0]
        extrema = sorted(
            set(
                [round(float(x), 2) for x in maxima[-20:]]
                + [round(float(x), 2) for x in minima[-20:]]
            )
        )
        if not extrema:
            extrema = [round(float(np.mean(half)), 2)]
        attractors.append(extrema)

    return {"param_name": param_name, "param_values": param_values, "attractors": attractors}


def sensitivity_analysis(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_names: list[str],
    perturbation: float = 0.1,
) -> dict[str, Any]:
    """Compute firing rate sensitivity to each parameter (±perturbation fraction)."""
    base_result = simulate_fn(**base_config)
    base_rate = base_result["stats"]["rate_hz"]
    sensitivities: list[dict[str, Any]] = []

    for pname in param_names:
        params = dict(base_config.get("params") or {})
        base_val = params.get(pname, 0.0)
        if base_val == 0:
            sensitivities.append({"param": pname, "sensitivity": 0.0, "base_rate": base_rate})
            continue

        delta = abs(base_val) * perturbation
        results = []
        for sign in [-1, 1]:
            cfg = dict(base_config)
            p = dict(params)
            p[pname] = base_val + sign * delta
            cfg["params"] = p
            r = simulate_fn(**cfg)
            results.append(r["stats"]["rate_hz"])

        deriv = (results[1] - results[0]) / (2 * delta) if delta > 0 else 0.0
        sensitivities.append(
            {
                "param": pname,
                "sensitivity": round(abs(deriv) * abs(base_val) / max(base_rate, 0.1), 4),
                "base_rate": base_rate,
                "rate_minus": results[0],
                "rate_plus": results[1],
            }
        )

    sensitivities.sort(key=lambda s: s["sensitivity"], reverse=True)
    return {"base_rate": base_rate, "sensitivities": sensitivities}


def nullclines_2d(
    equations: list[str],
    params: dict[str, float],
    var_names: list[str],
    ranges: dict[str, tuple[float, float]],
    grid_size: int = 80,
) -> dict[str, Any]:
    """Compute nullclines for a 2-variable ODE system on a grid."""
    from sc_neurocore.neurons.equation_builder import from_equations

    if len(var_names) < 2 or len(equations) < 2:
        return {"error": "Need 2+ variables for nullclines"}

    v0, v1 = var_names[0], var_names[1]
    r0 = ranges.get(v0, (-80, 40))
    r1 = ranges.get(v1, (-1, 1))

    x = np.linspace(r0[0], r0[1], grid_size)
    y = np.linspace(r1[0], r1[1], grid_size)
    X, Y = np.meshgrid(x, y)

    neuron = from_equations(*equations, params=params, init={v0: 0.0, v1: 0.0}, dt=0.01)

    dv0 = np.zeros_like(X)
    dv1 = np.zeros_like(X)

    compiled_eq0 = neuron._compiled_eqs[v0]
    compiled_eq1 = neuron._compiled_eqs[v1]

    for i in range(grid_size):
        for j in range(grid_size):
            env = dict(neuron._namespace)
            env.update(neuron.parameters)
            env.update(neuron.constants)
            env[v0] = float(X[i, j])
            env[v1] = float(Y[i, j])
            env["I"] = 0.0
            try:
                # Bandit B307 justification: `compiled_eq0/1` come from
                # `EquationNeuron._compiled_eqs`, which has already
                # passed the AST whitelist in `_validate_expr` (no
                # imports, no attribute escapes). Empty `__builtins__`
                # blocks the residual escape vectors.
                dv0[i, j] = float(eval(compiled_eq0, {"__builtins__": {}}, env))  # nosec B307
                dv1[i, j] = float(eval(compiled_eq1, {"__builtins__": {}}, env))  # nosec B307
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

    # Extract zero-contours via sign changes
    def contour_points(Z: np.ndarray[Any, Any], threshold: float = 0.0) -> list[list[float]]:
        pts = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                vals = [Z[i, j], Z[i + 1, j], Z[i, j + 1], Z[i + 1, j + 1]]
                if min(vals) <= threshold <= max(vals):
                    pts.append([float(X[i, j]), float(Y[i, j])])
        return pts

    nc0 = contour_points(dv0)
    nc1 = contour_points(dv1)

    return {
        "var_names": [v0, v1],
        "nullcline_0": {"variable": v0, "points": nc0},
        "nullcline_1": {"variable": v1, "points": nc1},
    }


def heatmap_2d(
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
) -> dict[str, Any]:
    """Sweep two parameters and compute firing rate heatmap."""
    x_vals = np.linspace(x_min, x_max, x_steps).tolist()
    y_vals = np.linspace(y_min, y_max, y_steps).tolist()
    rates = np.zeros((y_steps, x_steps))
    failures: list[dict[str, Any]] = []

    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            cfg = dict(base_config)
            params = dict(cfg.get("params") or {})
            params[param_x] = xv
            params[param_y] = yv
            cfg["params"] = params
            try:
                result = simulate_fn(**cfg)
                rates[j, i] = result["stats"]["rate_hz"]
            except Exception as exc:
                failures.append(
                    {
                        "grid_index": [j, i],
                        "param_x_value": float(xv),
                        "param_y_value": float(yv),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    total_points = x_steps * y_steps
    if failures:
        raise ValueError(
            f"heatmap sweep failed for {len(failures)}/{total_points} points",
            {
                "failed_points": len(failures),
                "total_points": total_points,
                "failure_rate": float(len(failures)) / float(max(total_points, 1)),
                "failures": failures,
            },
        )

    return {
        "param_x": param_x,
        "x_values": x_vals,
        "param_y": param_y,
        "y_values": y_vals,
        "rates": rates.tolist(),
        "rate_min": float(np.min(rates)),
        "rate_max": float(np.max(rates)),
        "failed_points": 0,
        "total_points": total_points,
        "failure_rate": 0.0,
    }


def spike_triggered_average(
    time: list[float],
    voltage: list[float],
    spikes: list[int],
    dt: float,
    window_ms: float = 20.0,
) -> dict[str, Any]:
    """Compute spike-triggered average of voltage around each spike."""
    if len(spikes) < 2:
        return {"time_ms": [], "average": [], "n_spikes": len(spikes)}

    half_win = int(window_ms / dt / 2)
    if half_win < 1:
        half_win = 1

    v = np.array(voltage)
    snippets = []
    for idx in spikes:
        lo = idx - half_win
        hi = idx + half_win
        if lo >= 0 and hi < len(v):
            snippets.append(v[lo:hi])

    if not snippets:
        return {"time_ms": [], "average": [], "n_spikes": 0}

    avg = np.mean(snippets, axis=0)
    t_ms = (np.arange(len(avg)) - half_win) * dt

    return {
        "time_ms": t_ms.tolist(),
        "average": avg.tolist(),
        "n_spikes": len(snippets),
    }


def frequency_response(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    freq_min: float = 1.0,
    freq_max: float = 100.0,
    n_freqs: int = 20,
    amplitude: float = 10.0,
) -> dict[str, Any]:
    """Sweep sinusoidal current frequency and measure spike rate response."""
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs).tolist()
    rates: list[float] = []

    for freq in freqs:
        cfg = dict(base_config)
        dt = cfg.get("dt", 0.1)
        duration = cfg.get("duration", 200.0)
        n_steps = min(int(duration / dt), 100_000)

        result = simulate_fn(
            **{
                **cfg,
                "current": amplitude,
                "protocol": "sine",
                "frequency_hz": freq,
            }
        )
        rates.append(result["stats"]["rate_hz"])

    return {"frequencies_hz": freqs, "rates": rates, "amplitude": amplitude}


def precision_compare(
    equations: list[str],
    threshold: str | None,
    reset: str | None,
    params: dict[str, float] | None,
    init: dict[str, float] | None,
    dt: float,
    duration: float,
    current: float,
) -> dict[str, Any]:
    """Compare float64 vs Q8.8 fixed-point simulation of the same ODE."""
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
    def q88(val: float) -> float:
        quantized = round(val * 256) / 256
        return max(-128.0, min(127.996, quantized))

    q_params = {k: q88(v) for k, v in (params or {}).items()}
    q_init = {k: q88(v) for k, v in (init or {}).items()}

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
    float_v = np.array(float_result["states"][var0])
    fixed_v = np.array(fixed_result["states"][var0])
    n = min(len(float_v), len(fixed_v))
    error = np.abs(float_v[:n] - fixed_v[:n])

    return {
        "float_result": float_result,
        "fixed_result": fixed_result,
        "error": {
            "variable": var0,
            "max_error": round(float(np.max(error)), 6),
            "mean_error": round(float(np.mean(error)), 6),
            "rms_error": round(float(np.sqrt(np.mean(error**2))), 6),
            "trace": error.tolist(),
        },
        "quantized_params": q_params,
    }
