# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research analysis functions for Studio

"""Parameter sweeps and response curves over Studio simulation runs.

Every function returns its numbers next to a
:class:`~sc_neurocore.studio.analysis_contract.MetricContract` stating the
definition, units, applicability and limits of what was computed. Where a
metric is undefined at a point (no spikes to normalise by, a zero parameter
under a relative perturbation, too few samples for an attractor) the point is
reported as undefined with its reason rather than as a zero.

The nullcline and precision analyses live in their own modules
(:mod:`sc_neurocore.studio.nullclines`,
:mod:`sc_neurocore.studio.precision_compare`) and are re-exported here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sc_neurocore.studio.analysis_contract import (
    MODEL_DEFINED_UNIT,
    MetricContract,
    attach_contract,
)
from sc_neurocore.studio.nullclines import nullclines_2d
from sc_neurocore.studio.precision_compare import precision_compare
from sc_neurocore.studio.trace_projection import full_state_traces

SENSITIVITY_PERTURBATION = 0.1
ATTRACTOR_MIN_SAMPLES = 10
ATTRACTOR_EXTREMA_KEPT = 20
ATTRACTOR_DECIMALS = 2

RATE_DEFINITION = (
    "spike count divided by the simulated duration (Hz, time step in ms); the "
    "transient from the initial state is included, no settling window is discarded"
)


def _rate_units() -> dict[str, str]:
    return {"rates": "Hz", "currents": MODEL_DEFINED_UNIT, "current": MODEL_DEFINED_UNIT}


def fi_curve_sweep(
    simulate_fn: Callable[..., dict[str, Any]],
    i_min: float,
    i_max: float,
    i_steps: int,
) -> dict[str, Any]:
    """Sweep a constant current and report the firing rate at each level.

    Parameters
    ----------
    simulate_fn:
        Callable accepting ``current=`` and returning a run result with
        ``stats.rate_hz``.
    i_min, i_max, i_steps:
        Inclusive current range and number of levels.

    Returns
    -------
    dict
        ``currents``, ``rates`` and the metric contract.
    """
    currents = np.linspace(i_min, i_max, i_steps).tolist()
    rates = [float(simulate_fn(current=float(level))["stats"]["rate_hz"]) for level in currents]
    payload: dict[str, Any] = {"currents": currents, "rates": rates}
    return attach_contract(
        payload,
        MetricContract(
            kind="fi-curve",
            definition=RATE_DEFINITION,
            units=_rate_units(),
            applicability=(
                "constant-current protocol; each level starts from the same initial state",
            ),
            limitations=(
                "rate resolution is one spike per simulated duration",
                "a rate of 0 is a measured absence of spikes within the duration, not a "
                "statement about the rheobase",
            ),
        ),
    )


def bifurcation_sweep(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_name: str,
    param_min: float,
    param_max: float,
    n_values: int = 30,
    *,
    variable: str | None = None,
) -> dict[str, Any]:
    """Sweep one parameter and record the late-run extrema of one state trace.

    This is a numerical extrema sweep under the configured drive, not a
    bifurcation continuation: no equilibrium branch is followed and no
    stability is computed. At each parameter value the second half of the
    trace is inspected; its local maxima and minima (up to
    :data:`ATTRACTOR_EXTREMA_KEPT` of each, rounded to
    :data:`ATTRACTOR_DECIMALS`) are reported as the attractor sample. A
    trace without extrema reports its mean as a single fixed-point sample; a
    trace with fewer than :data:`ATTRACTOR_MIN_SAMPLES` late samples is
    reported as insufficient.

    Parameters
    ----------
    simulate_fn:
        Callable accepting ``params=`` (and the other base-config keys) and
        returning a run result with raw traces.
    base_config:
        Run configuration shared by every sweep point (``protocol`` is
        reported).
    param_name, param_min, param_max, n_values:
        Swept parameter and its inclusive range.
    variable:
        State variable analysed; the first raw trace when ``None``.

    Returns
    -------
    dict
        ``param_name``, ``param_values``, ``attractors`` (one list per
        value), ``attractor_kinds`` (``extrema`` / ``fixed_point`` /
        ``insufficient_samples``), ``variable``, ``protocol`` and the
        metric contract.
    """
    param_values = np.linspace(param_min, param_max, n_values).tolist()
    attractors: list[list[float]] = []
    kinds: list[str] = []
    analysed: str | None = variable

    for value in param_values:
        cfg = dict(base_config)
        params = dict(cfg.get("params") or {})
        params[param_name] = value
        cfg["params"] = params

        result = simulate_fn(**cfg)
        traces = full_state_traces(result)
        if analysed is None:
            analysed = next(iter(traces))
        if analysed not in traces:
            raise ValueError(f"variable {analysed!r} is not a state trace of the run")
        trace = traces[analysed]
        half = trace[len(trace) // 2 :]
        if len(half) < ATTRACTOR_MIN_SAMPLES:
            attractors.append([])
            kinds.append("insufficient_samples")
            continue
        arr = np.array(half, dtype=np.float64)
        diffs = np.diff(np.sign(np.diff(arr)))
        maxima = arr[1:-1][diffs < 0]
        minima = arr[1:-1][diffs > 0]
        extrema = sorted(
            set(
                [round(float(x), ATTRACTOR_DECIMALS) for x in maxima[-ATTRACTOR_EXTREMA_KEPT:]]
                + [round(float(x), ATTRACTOR_DECIMALS) for x in minima[-ATTRACTOR_EXTREMA_KEPT:]]
            )
        )
        if extrema:
            attractors.append(extrema)
            kinds.append("extrema")
        else:
            attractors.append([round(float(np.mean(arr)), ATTRACTOR_DECIMALS)])
            kinds.append("fixed_point")

    protocol = str(base_config.get("protocol", "constant"))
    payload: dict[str, Any] = {
        "param_name": param_name,
        "param_values": param_values,
        "attractors": attractors,
        "attractor_kinds": kinds,
        "variable": analysed,
        "protocol": protocol,
    }
    return attach_contract(
        payload,
        MetricContract(
            kind="numerical-extrema-sweep",
            definition=(
                "local maxima and minima of the analysed state trace over the second half "
                f"of each run (last {ATTRACTOR_EXTREMA_KEPT} of each, rounded to "
                f"{ATTRACTOR_DECIMALS} decimals); a trace without extrema reports its mean"
            ),
            units={"param_values": MODEL_DEFINED_UNIT, "attractors": MODEL_DEFINED_UNIT},
            applicability=(
                f"drive protocol {protocol!r}; the extrema reflect the driven response, "
                "not an autonomous attractor",
                "each run restarts from the same initial state",
            ),
            limitations=(
                "not a bifurcation continuation: no branch following, no stability, no "
                "detection of bifurcation points",
                "the transient is assumed to end within the first half of the run",
            ),
        ),
    )


def sensitivity_analysis(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_names: Sequence[str],
    perturbation: float = SENSITIVITY_PERTURBATION,
) -> dict[str, Any]:
    """Rate elasticity of each parameter under a symmetric relative perturbation.

    The elasticity is ``|rate(p + δ) − rate(p − δ)| / (2 δ) · |p| / rate(p)``
    with ``δ = perturbation · |p|``. It is undefined (reported as ``null``
    with a reason) when the base rate is zero or when the parameter is zero,
    because a relative perturbation of zero is no perturbation.

    Parameters
    ----------
    simulate_fn:
        Callable accepting ``params=`` and returning a run result with
        ``stats.rate_hz``.
    base_config:
        Run configuration; ``params`` holds the base values.
    param_names:
        Parameters to perturb.
    perturbation:
        Relative perturbation of each parameter.

    Returns
    -------
    dict
        ``base_rate``, ``sensitivities`` (sorted, undefined last) and the
        metric contract; each row carries ``sensitivity`` (or ``null``),
        ``reason`` when undefined, ``rate_minus`` / ``rate_plus`` when
        computed.
    """
    base_result = simulate_fn(**base_config)
    base_rate = float(base_result["stats"]["rate_hz"])
    sensitivities: list[dict[str, Any]] = []
    undefined = 0

    for name in param_names:
        params = dict(base_config.get("params") or {})
        base_val = float(params.get(name, 0.0))
        if base_val == 0.0:
            sensitivities.append(
                {
                    "param": name,
                    "sensitivity": None,
                    "base_rate": base_rate,
                    "reason": "relative perturbation of a zero parameter is no perturbation",
                }
            )
            undefined += 1
            continue

        delta = abs(base_val) * perturbation
        rates: list[float] = []
        for sign in (-1.0, 1.0):
            cfg = dict(base_config)
            perturbed = dict(params)
            perturbed[name] = base_val + sign * delta
            cfg["params"] = perturbed
            rates.append(float(simulate_fn(**cfg)["stats"]["rate_hz"]))
        row: dict[str, Any] = {
            "param": name,
            "base_rate": base_rate,
            "rate_minus": rates[0],
            "rate_plus": rates[1],
        }
        if base_rate <= 0.0:
            row["sensitivity"] = None
            row["reason"] = "no spikes at the base configuration; the elasticity is undefined"
            undefined += 1
        else:
            derivative = (rates[1] - rates[0]) / (2.0 * delta)
            row["sensitivity"] = round(abs(derivative) * abs(base_val) / base_rate, 4)
        sensitivities.append(row)

    sensitivities.sort(key=lambda row: (row["sensitivity"] is None, -(row["sensitivity"] or 0.0)))
    defined = len(sensitivities) - undefined
    if not sensitivities or defined == len(sensitivities):
        domain = "complete"
    elif defined == 0:
        domain = "empty"
    else:
        domain = "partial"
    payload: dict[str, Any] = {"base_rate": base_rate, "sensitivities": sensitivities}
    return attach_contract(
        payload,
        MetricContract(
            kind="rate-elasticity",
            definition=(
                "|rate(p+δ) − rate(p−δ)| / (2δ) · |p| / rate(p) with δ = "
                f"{perturbation} · |p| (dimensionless central-difference elasticity of "
                "the firing rate)"
            ),
            units={
                "sensitivity": "dimensionless",
                "base_rate": "Hz",
                "rate_minus": "Hz",
                "rate_plus": "Hz",
            },
            applicability=(
                "base configuration with a non-zero firing rate and non-zero parameter values",
                "rates follow the f-I definition (transient included)",
            ),
            limitations=(
                "a finite-difference estimate at one perturbation size; a rate that is "
                "piecewise constant in the parameter yields 0 or a jump, not a derivative",
            ),
            domain=domain,  # type: ignore[arg-type]
            domain_detail={"undefined": undefined, "total": len(sensitivities)},
        ),
    )


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
    """Sweep two parameters and compute the firing-rate map.

    The sweep fails closed: when any grid point fails the whole request is
    rejected with every failure listed, so a partial map is never returned
    with silent zeros.
    """
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

    payload: dict[str, Any] = {
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
    return attach_contract(
        payload,
        MetricContract(
            kind="rate-map",
            definition=RATE_DEFINITION,
            units={"rates": "Hz", "x_values": MODEL_DEFINED_UNIT, "y_values": MODEL_DEFINED_UNIT},
            applicability=(
                f"drive protocol {base_config.get('protocol', 'constant')!r}; each grid "
                "point restarts from the same initial state",
            ),
            limitations=("rate resolution is one spike per simulated duration",),
        ),
    )


def spike_triggered_average(
    time: Sequence[float],
    voltage: Sequence[float],
    spikes: Sequence[int],
    dt: float,
    window_ms: float = 20.0,
) -> dict[str, Any]:
    """Average of the trace in a symmetric window around every complete spike.

    Spikes whose window would leave the trace are excluded and counted
    against ``n_spikes``; the window half-width is ``window_ms / 2`` rounded
    down to whole steps (at least one step).
    """
    if len(spikes) < 2:
        payload: dict[str, Any] = {"time_ms": [], "average": [], "n_spikes": len(spikes)}
        return attach_contract(payload, _sta_contract(window_ms, "empty", len(spikes), 0))

    half_win = int(window_ms / dt / 2)
    if half_win < 1:
        half_win = 1

    v = np.array(voltage, dtype=np.float64)
    snippets = []
    for idx in spikes:
        lo = idx - half_win
        hi = idx + half_win
        if lo >= 0 and hi < len(v):
            snippets.append(v[lo:hi])

    if not snippets:
        payload = {"time_ms": [], "average": [], "n_spikes": 0}
        return attach_contract(payload, _sta_contract(window_ms, "empty", len(spikes), 0))

    avg = np.mean(snippets, axis=0)
    t_ms = (np.arange(len(avg)) - half_win) * dt
    domain = "complete" if len(snippets) == len(spikes) else "partial"
    payload = {"time_ms": t_ms.tolist(), "average": avg.tolist(), "n_spikes": len(snippets)}
    return attach_contract(payload, _sta_contract(window_ms, domain, len(spikes), len(snippets)))


def _sta_contract(window_ms: float, domain: str, spikes: int, used: int) -> MetricContract:
    return MetricContract(
        kind="spike-triggered-average",
        definition=(
            f"mean of the trace over a ±{window_ms / 2} ms window around each spike whose "
            "window lies inside the trace"
        ),
        units={"time_ms": "ms", "average": MODEL_DEFINED_UNIT},
        applicability=("at least two spikes with complete windows",),
        limitations=("spikes near the trace edges are excluded, not padded",),
        domain=domain,  # type: ignore[arg-type]
        domain_detail={"spikes": spikes, "windows_used": used},
    )


def frequency_response(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    freq_min: float = 1.0,
    freq_max: float = 100.0,
    n_freqs: int = 20,
    amplitude: float = 10.0,
) -> dict[str, Any]:
    """Sweep the frequency of a sinusoidal drive and record the firing rate.

    The drive is ``I(t) = amplitude · sin(2π f t)`` from the run's first
    step; the rate follows the f-I definition. Frequencies are spaced
    logarithmically.
    """
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs).tolist()
    rates: list[float] = []

    for freq in freqs:
        result = simulate_fn(
            **{
                **base_config,
                "current": amplitude,
                "protocol": "sine",
                "frequency_hz": freq,
            }
        )
        rates.append(float(result["stats"]["rate_hz"]))

    payload: dict[str, Any] = {"frequencies_hz": freqs, "rates": rates, "amplitude": amplitude}
    return attach_contract(
        payload,
        MetricContract(
            kind="frequency-response",
            definition=RATE_DEFINITION,
            units={"frequencies_hz": "Hz", "rates": "Hz", "amplitude": MODEL_DEFINED_UNIT},
            applicability=(
                "sinusoidal drive of the stated amplitude starting at phase 0",
                "the rate counts every spike in the run, so at low frequencies the number "
                "of drive cycles within the duration bounds the resolution",
            ),
            limitations=("no phase-locking or gain measure; rate only",),
        ),
    )


__all__ = [
    "ATTRACTOR_DECIMALS",
    "ATTRACTOR_EXTREMA_KEPT",
    "ATTRACTOR_MIN_SAMPLES",
    "RATE_DEFINITION",
    "SENSITIVITY_PERTURBATION",
    "bifurcation_sweep",
    "fi_curve_sweep",
    "frequency_response",
    "heatmap_2d",
    "nullclines_2d",
    "precision_compare",
    "sensitivity_analysis",
    "spike_triggered_average",
]
