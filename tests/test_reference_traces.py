# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace neuron validation contracts

"""Production contracts for the neuron reference-trace validation harness."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceSpec,
    list_reference_trace_specs,
    load_reference_trace_spec,
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas

_STOCHASTIC_SCHEMA_NAMES = frozenset({"escape_rate", "poisson"})
_DETERMINISTIC_SCHEMA_TRACES = {
    "adex": "adex_resting_adaptation_doi",
    "connor_stevens": "connor_stevens_driven_spiking_doi",
    "dpi_neuron": "dpi_neuron_driven_spiking_doi",
    "exp_if": "exp_if_resting_exponential_doi",
    "fitzhugh_nagumo": "fitzhugh_nagumo_driven_oscillation_doi",
    "fitzhugh_rinzel": "fitzhugh_rinzel_driven_bursting_doi",
    "glif": "glif_constant_current_threshold_adaptation",
    "hindmarsh_rose": "hindmarsh_rose_short_bursting_prefix",
    "hodgkin_huxley": "hodgkin_huxley_driven_spiking_doi",
    "izhikevich": "izhikevich_regular_spiking_doi",
    "izhikevich2007": "izhikevich2007_regular_spiking_doi",
    "lapicque": "lapicque_constant_current_closed_form",
    "lif": "lif_constant_current_closed_form",
    "mckean": "mckean_driven_oscillation_doi",
    "mihalas_niebur": "mihalas_niebur_driven_spiking_doi",
    "morris_lecar": "morris_lecar_driven_oscillation_doi",
    "pernarowski": "pernarowski_autonomous_bursting_doi",
    "terman_wang": "terman_wang_legion_oscillation_doi",
    "wilson_hr": "wilson_hr_driven_spiking_doi",
    "perfect_integrator": "perfect_integrator_constant_current_sawtooth",
    "quadratic_if": "quadratic_if_zero_current_analytic",
    "resonate_fire": "resonate_fire_subthreshold_resonance_doi",
    "rulkov_map": "rulkov_map_driven_spiking_doi",
    "theta": "theta_constant_current_phase_analytic",
    "wang_buzsaki": "wang_buzsaki_driven_spiking_doi",
}


def _summarise(recorded: dict[str, list[float]], spikes: list[int]) -> dict[str, float]:
    """Return the shared spike-count / first-spike-step / per-variable feature map.

    Every reference helper that tracks a per-step ``spikes`` list and one or more
    recorded state-variable trajectories reduces them to the same feature contract: a
    total spike count, the 1-indexed first-spike step (``-1`` when silent), and the
    final / minimum / maximum / mean of each recorded variable. Centralising the tail
    keeps the independent-parity helpers byte-identical in how they summarise, so a
    drift in one helper's reduction cannot silently diverge from the others.

    Parameters
    ----------
    recorded:
        Mapping from state-variable name to its per-step trajectory.
    spikes:
        Per-step spike indicators (``1`` on a spiking step, ``0`` otherwise).

    Returns
    -------
    dict of str to float
        The feature map keyed by ``spike_count``, ``first_spike_step``, and
        ``final.<var>`` / ``min.<var>`` / ``max.<var>`` / ``mean.<var>`` per variable.
    """
    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for variable, values in recorded.items():
        features[f"final.{variable}"] = values[-1]
        features[f"min.{variable}"] = min(values)
        features[f"max.{variable}"] = max(values)
        features[f"mean.{variable}"] = math.fsum(values) / len(values)
    return features


def _closed_form_features(
    *,
    initial: float,
    steady: float,
    tau: float,
    dt: float,
    steps: int,
) -> dict[str, float]:
    values = [
        steady + (initial - steady) * math.exp(-(step * dt) / tau) for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _quadratic_if_zero_current_features(*, dt: float, steps: int) -> dict[str, float]:
    values = [-1.0 / (1.0 + step * dt) for step in range(1, steps + 1)]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _perfect_integrator_sawtooth_features(
    *,
    current: float,
    dt: float,
    steps: int,
    c_m: float = 1.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> dict[str, float]:
    """Return exact post-reset features for constant-current perfect integration."""
    values: list[float] = []
    spikes: list[int] = []
    voltage = v_reset
    increment = current * dt / c_m
    for _ in range(steps):
        voltage += increment
        if voltage >= v_threshold:
            spikes.append(1)
            voltage = v_reset
        else:
            spikes.append(0)
        values.append(voltage)

    return _summarise({"v": values}, spikes)


def _theta_constant_current_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return continuous theta-neuron phase features for constant positive current."""
    if current <= 0.0:
        msg = "theta analytic helper requires positive current"
        raise ValueError(msg)
    root_current = math.sqrt(current)
    values = [
        2.0 * math.atan(root_current * math.tan(root_current * step * dt))
        for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.theta": values[-1],
        "min.theta": min(values),
        "max.theta": max(values),
        "mean.theta": math.fsum(values) / len(values),
    }


def _resonate_fire_linear_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact Euler features for the linear resonate-and-fire schema."""
    omega = 0.5
    damping = -0.1
    threshold = 1.0
    x = 0.0
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dx = damping * x - omega * y + current
        dy = omega * x + damping * y
        x_next = x + dt * dx
        y_next = y + dt * dy
        if x_next > threshold:
            spikes.append(1)
            x = 0.0
            y = 0.0
        else:
            spikes.append(0)
            x = x_next
            y = y_next
        x_values.append(x)
        y_values.append(y)

    return _summarise({"x": x_values, "y": y_values}, spikes)


def _glif_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold GLIF5 recurrence.

    The Allen Institute GLIF5 membrane, adaptive threshold, and two after-spike
    currents are linear, so the schema runner's simultaneous explicit-Euler update
    has an exact independent re-derivation. For a subthreshold constant current the
    threshold is never crossed and both after-spike currents stay quiescent at zero.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``theta``, ``i_asc1``, and ``i_asc2``
        state variables plus spike-count and first-spike-step features.
    """
    v_rest = -70.0
    v_reset = -70.0
    resistance = 1.0
    tau_m = 10.0
    theta_inf = -50.0
    a_theta = 0.01
    tau_theta = 100.0
    tau_asc1 = 10.0
    tau_asc2 = 200.0
    delta_theta = 2.0
    r_asc1 = 1.0
    r_asc2 = 0.5

    v = v_rest
    theta = theta_inf
    i_asc1 = 0.0
    i_asc2 = 0.0
    recorded: dict[str, list[float]] = {"v": [], "theta": [], "i_asc1": [], "i_asc2": []}
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + resistance * current + i_asc1 + i_asc2) / tau_m
        dtheta = (theta_inf - theta + a_theta * (v - v_rest)) / tau_theta
        di_asc1 = -i_asc1 / tau_asc1
        di_asc2 = -i_asc2 / tau_asc2
        v_next = v + dv * dt
        theta_next = theta + dtheta * dt
        i_asc1_next = i_asc1 + di_asc1 * dt
        i_asc2_next = i_asc2 + di_asc2 * dt
        if v_next > theta_next:
            spikes.append(1)
            v_next = v_reset
            theta_next = theta_next + delta_theta
            i_asc1_next = i_asc1_next + r_asc1
            i_asc2_next = i_asc2_next + r_asc2
        else:
            spikes.append(0)
        v, theta, i_asc1, i_asc2 = v_next, theta_next, i_asc1_next, i_asc2_next
        recorded["v"].append(v)
        recorded["theta"].append(theta)
        recorded["i_asc1"].append(i_asc1)
        recorded["i_asc2"].append(i_asc2)

    return _summarise(recorded, spikes)


def _izhikevich_rs_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the regular-spiking Izhikevich recurrence.

    The Izhikevich (2003) quadratic membrane and linear recovery equations are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane crosses the ``v > 30`` peak. The reference is therefore an independent
    re-derivation of the committed spike-bearing trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.02
    b = 0.2
    c = -65.0
    d = 8.0
    v = -65.0
    u = -14.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = 0.04 * v**2 + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next > 30:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)


def _izhikevich2007_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the Izhikevich 2007 recurrence.

    The Izhikevich (2007) biophysical quadratic membrane ``C dv/dt =
    k (v - vr) (v - vt) - u + I`` and linear recovery ``du/dt = a (b (v - vr) - u)``
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane reaches the ``v >= vpeak`` peak. The right-hand side is polynomial, so
    the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed regular-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    c_m = 100.0
    k = 0.7
    vr = -60.0
    vt = -40.0
    vpeak = 35.0
    a = 0.03
    b = -2.0
    c = -50.0
    d = 100.0
    v = -60.0
    u = 0.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (k * (v - vr) * (v - vt) - u + current) / c_m
        du = a * (b * (v - vr) - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next >= vpeak:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)


def _dpi_neuron_driven_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the driven DPI current-mode recurrence.

    The DYNAP-SE differential-pair-integrator membrane ``tau dI_mem/dt =
    -I_mem + gain * I_syn + I_leak`` (Chicca et al. 2014) is advanced with the same
    explicit-Euler update the schema runner applies, and the ``i_mem = i_reset`` reset
    fires whenever the post-update current reaches the ``i_mem >= i_threshold`` level.
    The right-hand side is linear, so the recurrence reproduces the schema runner
    bit-for-bit — an independent re-derivation of the committed driven-spiking trace,
    not a copy of the runner. The non-negative drive keeps ``i_mem`` non-negative, so
    the source model's ``max(i_mem, 0)`` current rectification is inert and correctly
    absent from this continuous update.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``i_mem`` state variable plus spike-count and
        first-spike-step features.
    """
    i_threshold = 1.0
    i_reset = 0.0
    i_leak = 0.01
    tau = 20.0
    gain = 1.0
    i_mem = 0.0
    i_mem_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        di = (-i_mem + gain * current + i_leak) / tau
        i_mem_next = i_mem + di * dt
        if i_mem_next >= i_threshold:
            spikes.append(1)
            i_mem_next = i_reset
        else:
            spikes.append(0)
        i_mem = i_mem_next
        i_mem_values.append(i_mem)

    return _summarise({"i_mem": i_mem_values}, spikes)


def _mihalas_niebur_driven_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact fourth-order Runge-Kutta features for the driven Mihalas-Niebur flow.

    The generalised integrate-and-fire flow (Mihalaş & Niebur 2009) advances four linear
    states — membrane ``dv/dt = (-(v - v_rest) + i1 + i2 + I) / tau_v``, adaptive threshold
    ``dtheta/dt = (theta_inf - theta + a (v - v_rest)) / tau_theta`` and two spike-triggered
    currents ``di1/dt = -i1 / tau_1``, ``di2/dt = -i2 / tau_2`` — with the classical RK4
    step the schema runner applies, and the adaptive reset ``v = v_reset + b (v - v_rest)``,
    ``theta = max(theta, theta_reset)``, ``i1 += r1``, ``i2 += r2`` fires whenever the
    post-step membrane reaches the state-to-state ``v >= theta`` threshold. Every derivative
    is linear, so the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed driven-spiking trace, not a copy of the runner. Because
    ``theta_reset`` (1.3) exceeds ``theta_inf`` (1.0) the max() threshold floor engages on
    every spike, so the state-to-state comparison is a genuine adaptive threshold.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``theta``, ``i1`` and ``i2`` state variables
        plus spike-count and first-spike-step features.
    """
    v_rest = 0.0
    v_reset = 0.0
    theta_reset = 1.3
    theta_inf = 1.0
    tau_v = 10.0
    tau_theta = 40.0
    tau_1 = 15.0
    tau_2 = 80.0
    a = 0.1
    b = 0.1
    r1 = 0.2
    r2 = -0.15
    v = 0.0
    theta = 1.0
    i1 = 0.0
    i2 = 0.0
    half_dt = 0.5 * dt
    v_values: list[float] = []
    theta_values: list[float] = []
    i1_values: list[float] = []
    i2_values: list[float] = []
    spikes: list[int] = []

    def deriv(vv: float, th: float, j1: float, j2: float) -> tuple[float, float, float, float]:
        return (
            (-(vv - v_rest) + j1 + j2 + current) / tau_v,
            (theta_inf - th + a * (vv - v_rest)) / tau_theta,
            -j1 / tau_1,
            -j2 / tau_2,
        )

    for _ in range(steps):
        k1 = deriv(v, theta, i1, i2)
        k2 = deriv(
            v + half_dt * k1[0],
            theta + half_dt * k1[1],
            i1 + half_dt * k1[2],
            i2 + half_dt * k1[3],
        )
        k3 = deriv(
            v + half_dt * k2[0],
            theta + half_dt * k2[1],
            i1 + half_dt * k2[2],
            i2 + half_dt * k2[3],
        )
        k4 = deriv(
            v + dt * k3[0],
            theta + dt * k3[1],
            i1 + dt * k3[2],
            i2 + dt * k3[3],
        )
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        i1 = i1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i2 = i2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if v >= theta:
            spikes.append(1)
            v = v_reset + b * (v - v_rest)
            theta = max(theta, theta_reset)
            i1 = i1 + r1
            i2 = i2 + r2
        else:
            spikes.append(0)
        v_values.append(v)
        theta_values.append(theta)
        i1_values.append(i1)
        i2_values.append(i2)

    return _summarise(
        {"v": v_values, "theta": theta_values, "i1": i1_values, "i2": i2_values}, spikes
    )


def _fitzhugh_nagumo_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven FitzHugh-Nagumo oscillator.

    The FitzHugh (1961) cubic membrane and linear recovery equations are advanced
    with the same four-stage RK4 step and rising-edge spike detection the faithful
    schema runner applies, with **no reset** — the re-enrolled model is a genuine
    relaxation oscillator whose spikes are upward ``v >= 1`` threshold crossings, not
    integrate-and-fire resets. The cube is written ``v * v * v`` (not ``v ** 3``) so
    it is the exact IEEE multiplication the runner and the hand model evaluate. The
    reference is an independent re-derivation of the committed relaxation-oscillation
    trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.7
    b = 0.8
    epsilon = 0.08
    threshold = 1.0
    v = -1.0
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + current,
            epsilon * (v_state + a - b * w_state),
        )

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _fitzhugh_rinzel_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven FitzHugh-Rinzel flow.

    The Rinzel (1987) three-state qualitative burster extends the FitzHugh-Nagumo
    fast subsystem with the ultra-slow ``y`` modulation equation. This independent
    recurrence advances all three coupled equations with one simultaneous four-stage
    RK4 step, then applies the maintained rising-edge ``v >= 1`` crossing decision
    without resetting any state. The cube is written ``v * v * v`` to reproduce the
    exact IEEE operation order of the hand model and schema runner; the recurrence is
    re-derived here rather than calling either implementation.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v``, ``w``, and ``y``, plus the spike count and
        first-spike step.
    """
    a = 0.7
    b = 0.8
    c = -0.775
    d = 1.0
    delta = 0.08
    mu = 0.0001
    threshold = 1.0
    v = -1.0
    w = -0.5
    y = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, y_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + y_state + current,
            delta * (a + v_state - b * w_state),
            mu * (c - v_state - d * y_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, y)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            y + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            y + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], y + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        y = y + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        y_values.append(y)

    return _summarise({"v": v_values, "w": w_values, "y": y_values}, spikes)


def _pernarowski_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the autonomous Pernarowski flow.

    The Pernarowski (1994) beta-cell model couples a fast cubic coordinate to
    recovery ``w`` and ultra-slow adaptation ``z``. This independent recurrence
    advances all three equations simultaneously with classical four-stage RK4,
    then applies the maintained rising-edge ``v >= 0.5`` crossing decision
    without resetting state. It is re-derived here rather than calling the hand
    model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v``, ``w``, and ``z``, plus the spike count
        and first-spike step.
    """
    alpha = 0.1
    beta = 0.5
    eps1 = 0.1
    eps2 = 0.001
    gamma = 0.5
    threshold = 0.5
    v = -1.0
    w = 0.0
    z = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, z_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state - z_state + current,
            eps1 * (v_state - gamma * w_state + alpha),
            eps2 * (beta * (v_state + 0.7) - z_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, z)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], z + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        z = z + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        z_values.append(z)

    return _summarise({"v": v_values, "w": w_values, "z": z_values}, spikes)


def _terman_wang_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Terman-Wang LEGION oscillator.

    This independent recurrence re-derives the maintained two-state Terman-Wang
    (1995) cubic fast nullcline and ``tanh``-gated slow recovery equation. It
    advances both states simultaneously through four Runge-Kutta stages, then
    applies the no-reset rising-edge ``v >= 1.5`` crossing decision without
    calling the hand model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v`` and ``w``, plus the spike count and
        first-spike step.
    """
    alpha = 3.0
    beta = 0.2
    epsilon = 0.02
    rho = 0.0
    threshold = 1.5
    v = -1.5
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        fast = 3.0 * v_state - v_state * v_state * v_state + 2.0
        recovery = alpha * (1.0 + math.tanh(v_state / beta))
        return fast - w_state + current + rho, epsilon * (recovery - w_state)

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w)
        k2 = deriv(v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], w + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _wilson_hr_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Wilson-HR cortical model.

    This independent recurrence re-derives Wilson's two-state polynomial flow,
    advances ``v`` and ``r`` simultaneously through four Runge-Kutta stages, and
    applies the level ``v >= 0.4`` spike decision. A spike hard-resets only ``v``
    to ``-0.7``; the RK4 candidate recovery state is preserved. The helper does not
    call the hand model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for post-reset ``v`` and candidate ``r``, plus the
        spike count and first-spike step.
    """
    tau_r = 1.9
    threshold = 0.4
    reset_voltage = -0.7
    v = -0.7
    r = 0.1
    v_values: list[float] = []
    r_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, r_state: float) -> tuple[float, float]:
        membrane = -(17.81 + 47.71 * v_state + 32.63 * v_state * v_state) * (v_state - 0.55)
        recovery_coupling = -26.0 * r_state * (v_state + 0.92)
        return (
            membrane + recovery_coupling + current,
            (-r_state + 1.35 * v_state + 1.03) / tau_r,
        )

    for _ in range(steps):
        k1 = deriv(v, r)
        k2 = deriv(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], r + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        r = r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spike = int(v >= threshold)
        if spike:
            v = reset_voltage
        spikes.append(spike)
        v_values.append(v)
        r_values.append(r)

    return _summarise({"v": v_values, "r": r_values}, spikes)


def _mckean_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven McKean oscillator.

    The McKean (1970) piecewise-linear FitzHugh-Nagumo caricature replaces the cubic
    membrane nullcline with the three-branch function ``f(v) = min(max(-v, v - a),
    1 - v)`` (min/max are supported by the schema DSL). The membrane and linear
    recovery equations are advanced with the same four-stage RK4 step and rising-edge
    ``v >= v_peak`` crossing detection the faithful schema runner applies, with **no
    reset** — the enrolled operating point (``epsilon = 0.2``, ``gamma = 0.5``,
    ``I = 0.6``) is a sustained relaxation oscillator whose spikes are upward threshold
    crossings. The right-hand side is exact arithmetic (comparisons and linear pieces,
    no cube or transcendental), so the reference is an independent re-derivation of the
    committed trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.25
    epsilon = 0.2
    gamma = 0.5
    v_peak = 0.8
    v = 0.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        f_v = min(max(-v_state, v_state - a), 1.0 - v_state)
        return f_v - w_state + current, epsilon * (v_state - gamma * w_state)

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        spikes.append(1 if (v >= v_peak and v_prev < v_peak) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _adex_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold AdEx recurrence.

    The Brette-Gerstner (2005) exponential membrane and linear adaptation equations
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies. For the resting zero-current protocol the ``v > -50`` threshold is never
    reached, so the ``v = v_reset``, ``w = w + b`` reset stays inactive and the
    reference is an independent re-derivation of the committed quiet trajectory.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    v_rest = -65.0
    v_reset = -68.0
    v_rh = -55.0
    delta_t = 2.0
    tau = 20.0
    tau_w = 100.0
    a = 0.5
    b_adapt = 7.0
    capacitance = 200.0
    v = -65.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + delta_t * math.exp((v - v_rh) / delta_t)) / tau + (
            -w + current
        ) / capacitance
        dw = (a * (v - v_rest) - w) / tau_w
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > -50:
            spikes.append(1)
            v_next = v_reset
            w_next = w_next + b_adapt
        else:
            spikes.append(0)
        v, w = v_next, w_next
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _exp_if_subthreshold_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact explicit-Euler features for the resting exponential-IF recurrence.

    The Fourcaud-Trocme (2003) exponential integrate-and-fire membrane equation is
    advanced with the same explicit-Euler update the schema runner applies. For the
    resting zero-current protocol the ``v > 20`` peak is never reached, so the
    ``v = v_reset`` reset stays inactive and the reference is an independent
    re-derivation of the committed quiet trajectory.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` state variable plus spike-count and
        first-spike-step features.
    """
    v_rest = -70.0
    v_reset = -70.0
    v_threshold = -50.0
    v_peak = 20.0
    delta_t = 2.0
    tau_m = 10.0
    resistance = 1.0
    v = -70.0
    v_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (
            -(v - v_rest) + delta_t * math.exp((v - v_threshold) / delta_t) + resistance * current
        ) / tau_m
        v_next = v + dv * dt
        if v_next > v_peak:
            spikes.append(1)
            v_next = v_reset
        else:
            spikes.append(0)
        v = v_next
        v_values.append(v)

    return _summarise({"v": v_values}, spikes)


def _hindmarsh_rose_prefix_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact explicit-Euler features for the Hindmarsh-Rose bursting prefix.

    The Hindmarsh-Rose (1984) cubic fast subsystem and slow adaptation variable are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies. The schema reset is the identity map, and the committed short prefix
    stays below the ``x > 1`` threshold, so the reference is an independent
    re-derivation of the committed pre-bursting trajectory.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``x``, ``y``, and ``z`` state variables plus
        spike-count and first-spike-step features.
    """
    b = 3.0
    r = 0.001
    s = 4.0
    x_rest = -1.6
    x = -1.6
    y = -10.0
    z = 2.0
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dx = y - x**3 + b * x**2 - z + current
        dy = 1 - 5 * x**2 - y
        dz = r * (s * (x - x_rest) - z)
        x_next = x + dx * dt
        y_next = y + dy * dt
        z_next = z + dz * dt
        if x_next > 1.0:
            spikes.append(1)
        else:
            spikes.append(0)
        x, y, z = x_next, y_next, z_next
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return _summarise({"x": x_values, "y": y_values, "z": z_values}, spikes)


def _morris_lecar_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven Morris-Lecar oscillator.

    The Morris-Lecar (1981) calcium-potassium oscillator is the faithful
    conductance model: a genuine relaxation oscillator whose spikes are upward
    ``v >= v_threshold`` crossings, integrated with the same four-stage classical
    RK4 step the maintained ``MorrisLecarNeuron`` uses, with **no reset**. The
    sigmoidal calcium activation and potassium gating rate functions are transcribed
    verbatim from the schema, reusing ``numpy.tanh`` and ``numpy.cosh`` so the
    recurrence reproduces the schema runner bit-for-bit (the input current enters at
    every RK4 stage). The reference is an independent re-derivation of the committed
    driven-oscillation trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    c_m = 20.0
    g_ca = 4.0
    g_k = 8.0
    g_l = 2.0
    e_ca = 120.0
    e_k = -84.0
    e_l = -60.0
    v1 = -1.2
    v2 = 18.0
    v3 = 12.0
    v4 = 17.4
    phi = 0.06666666666666667
    v_threshold = 0.0
    v = -60.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        dv = (
            -g_ca * 0.5 * (1 + float(np.tanh((v_state - v1) / v2))) * (v_state - e_ca)
            - g_k * w_state * (v_state - e_k)
            - g_l * (v_state - e_l)
            + current
        ) / c_m
        dw = (
            phi
            * float(np.cosh((v_state - v3) / (2 * v4)))
            * (0.5 * (1 + float(np.tanh((v_state - v3) / v4))) - w_state)
        )
        return dv, dw

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _np_exp(x: float) -> float:
    """Return ``exp(x)`` through the same numpy implementation the schema runner uses.

    Parameters
    ----------
    x:
        Exponent argument.

    Returns
    -------
    float
        ``numpy.exp(x)`` as a Python float, bit-identical to the runner's rate terms.
    """
    return float(np.exp(x))


def _reference_exprel(x: float) -> float:
    """Return ``exprel(x) = (exp(x) - 1) / x`` with the removable-singularity limit.

    Mirrors ``EquationNeuron``'s vectorised ``exprel`` bit-for-bit: the ``|x| < 1e-9``
    branch returns the ``exprel(0) = 1`` limit as ``1 + x / 2``, and the regular
    branch uses ``numpy.expm1`` so conductance rate functions written as
    ``a / exprel(...)`` reproduce the runner exactly.

    Parameters
    ----------
    x:
        Rate-function argument.

    Returns
    -------
    float
        The exprel value matching the schema runner.
    """
    if abs(x) < 1e-9:
        return 1.0 + x / 2.0
    return float(np.expm1(x)) / x


def _hodgkin_huxley_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Hodgkin-Huxley oscillator.

    The Hodgkin-Huxley (1952) model is the faithful representation of the maintained
    ``HodgkinHuxleyNeuron(integrator="rk4")``, whose ``step()`` is itself a 100-sub-step
    macro step: each macro step advances ``substeps`` inner four-stage classical RK4
    sub-steps of ``dt`` over the same simultaneous derivative, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary, with **no reset**. The four-state membrane and Na/K
    gating rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` (the exprel-rewritten ``alpha_m`` /
    ``alpha_n``) so the recurrence reproduces the schema runner bit-for-bit. The
    reference is an independent re-derivation of the committed driven-spiking trace, not a
    copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, and ``n`` state variables
        plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 36.0
    g_l = 0.3
    e_na = 50.0
    e_k = -77.0
    e_l = -54.4
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n = sv
        dv = (
            -g_na * m**3 * h * (v - e_na) - g_k * n**4 * (v - e_k) - g_l * (v - e_l) + current
        ) / c_m
        dm = 1.0 / _reference_exprel(-(v + 40) / 10) * (1 - m) - 4 * _np_exp(-(v + 65) / 18) * m
        dh = 0.07 * _np_exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 35) / 10)) * h
        dn = 0.1 / _reference_exprel(-(v + 55) / 10) * (1 - n) - 0.125 * _np_exp(-(v + 65) / 80) * n
        return dv, dm, dh, dn

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(4))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(4))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(4))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(4))

    state: tuple[float, ...] = (-65.0, 0.05, 0.6, 0.32)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)


def _connor_stevens_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Connor-Stevens oscillator.

    The Connor-Stevens (1971) A-current model is the faithful representation of the
    maintained ``ConnorStevensNeuron`` (RK4, sub-stepped): each macro step advances
    ``substeps`` inner four-stage classical RK4 sub-steps of ``dt``, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary — matching the hand model's 100-sub-step-per-millisecond
    macro step and ``EquationNeuron.step``'s macro crossing, with **no reset**. The
    six-state membrane and Na/K/A-type gating rate functions are transcribed verbatim from
    the schema, reusing :func:`_np_exp` and :func:`_reference_exprel` (and the cube-root
    ``a``-gate) so the recurrence reproduces the schema runner bit-for-bit. The reference is
    an independent re-derivation of the committed driven-spiking trace, not a copy of the
    runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, ``n``, ``a``, and ``b``
        state variables plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 20.0
    g_a = 47.7
    g_l = 0.3
    e_na = 55.0
    e_k = -72.0
    e_a = -75.0
    e_l = -17.0
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": [], "a": [], "b": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n, a, b = sv
        dv = (
            -g_na * m**3 * h * (v - e_na)
            - g_k * n**4 * (v - e_k)
            - g_a * a**3 * b * (v - e_a)
            - g_l * (v - e_l)
            + current
        ) / c_m
        dm = (
            3.8 / _reference_exprel(-(v + 29.7) / 10) * (1 - m)
            - 15.2 * _np_exp(-(v + 54.7) / 18) * m
        )
        dh = 0.266 * _np_exp(-(v + 48) / 20) * (1 - h) - 3.8 / (1 + _np_exp(-(v + 18) / 10)) * h
        dn = (
            0.2 / _reference_exprel(-(v + 45.7) / 10) * (1 - n)
            - 0.25 * _np_exp(-(v + 55.7) / 80) * n
        )
        da = (
            (0.0761 * _np_exp((v + 94.22) / 31.84) / (1 + _np_exp((v + 1.17) / 28.93)))
            ** (1.0 / 3.0)
            - a
        ) / (0.3632 + 1.158 / (1 + _np_exp((v + 55.96) / 20.12)))
        db = (1 / (1 + _np_exp((v + 53.3) / 14.54)) ** 4 - b) / (
            1.24 + 2.678 / (1 + _np_exp((v + 50) / 16.027))
        )
        return dv, dm, dh, dn, da, db

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(6))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(6))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(6))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(6))

    state: tuple[float, ...] = (-68.0, 0.01, 0.99, 0.1, 0.5, 0.1)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n", "a", "b")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)


def _wang_buzsaki_macrostep_gauss_seidel_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step Gauss-Seidel features for the driven Wang-Buzsaki oscillator.

    The Wang-Buzsaki (1996) fast-spiking interneuron is the faithful representation of the
    maintained ``WangBuzsakiNeuron``: each macro step advances ``substeps`` inner sequential
    (Gauss-Seidel) forward-Euler sub-steps of ``dt`` — the gating variables ``h`` and ``n``
    are updated from the old voltage first, then the membrane voltage ``v`` from the
    already-updated gates (the schema declares ``method="gauss_seidel"`` with state ordered
    ``h, n, v``). Sodium activation is instantaneous: ``m_inf = alpha_m/(alpha_m+beta_m)``
    with ``alpha_m = 1/exprel(-(v+35)/10)`` (the exprel rewrite of ``0.1*(v+35)/(1-exp(...))``)
    and ``beta_m = 4*exp(-(v+60)/18)``; the potassium rate ``alpha_n`` is likewise
    ``0.1/exprel(-(v+34)/10)``. The rising-edge ``v >= v_threshold`` crossing is evaluated
    only on the macro boundary against the condition at the previous macro boundary, with
    **no reset**. The rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` so the recurrence reproduces the schema
    runner bit-for-bit. The reference is an independent re-derivation of the committed
    driven-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner Gauss-Seidel sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``h``, ``n``, and ``v`` state variables plus
        spike-count and first-spike-step features.
    """
    phi = 5.0
    g_na = 35.0
    g_k = 9.0
    g_l = 0.1
    e_na = 55.0
    e_k = -90.0
    e_l = -65.0
    capacitance = 1.0
    v_threshold = -20.0
    h = 0.8
    n = 0.1
    v = -65.0
    recorded: dict[str, list[float]] = {"h": [], "n": [], "v": []}
    spikes: list[int] = []
    for _ in range(steps):
        v_prev = v
        for _ in range(substeps):
            # ``h`` (declared first): reads the old voltage and old ``h``.
            h = (
                h
                + phi
                * (0.07 * _np_exp(-(v + 58) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 28) / 10)) * h)
                * dt
            )
            # ``n`` (declared second): reads the old voltage and old ``n``.
            n = (
                n
                + phi
                * (
                    0.1 / _reference_exprel(-(v + 34) / 10) * (1 - n)
                    - 0.125 * _np_exp(-(v + 44) / 80) * n
                )
                * dt
            )
            # ``v`` (declared last): reads the already-updated ``h``/``n`` and old ``v``.
            inv_exprel = 1 / _reference_exprel(-(v + 35) / 10)
            m_inf = inv_exprel / (inv_exprel + 4 * _np_exp(-(v + 60) / 18))
            v = (
                v
                + (
                    -g_na * m_inf**3 * h * (v - e_na)
                    - g_k * n**4 * (v - e_k)
                    - g_l * (v - e_l)
                    + current
                )
                / capacitance
                * dt
            )
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        recorded["h"].append(h)
        recorded["n"].append(n)
        recorded["v"].append(v)

    return _summarise(recorded, spikes)


def _rulkov_map_features(*, current: float, steps: int) -> dict[str, float]:
    """Return exact features for the Rulkov 2002 piecewise map iteration.

    The Rulkov (2002) fast/slow model is a discrete map, so an independent
    implementation of its three-branch fast map (rational subthreshold, spike
    plateau, hard reset) and slow drift reproduces the runner exactly — a map has no
    integration error, so independent parity is exact ground truth. Upward-crossing
    detection (post-update ``x >= 0`` with pre-update ``x < 0``) matches the hand
    model and schema runner.

    Parameters
    ----------
    current:
        Constant drive applied at every iteration.
    steps:
        Number of map iterations to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``x`` and ``y`` state variables plus
        spike-count and first-spike-step features.
    """
    alpha = 4.0
    sigma = -1.6
    mu = 0.001
    x = -1.0
    y = -3.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        x_previous = x
        if x <= 0:
            x_next = alpha / (1.0 - x) + y + current
        elif x < alpha + y + current:
            x_next = alpha + y + current
        else:
            x_next = -1.0
        y_next = y - mu * (x + 1.0) + mu * sigma
        x, y = x_next, y_next
        spikes.append(1 if x >= 0.0 and x_previous < 0.0 else 0)
        x_values.append(x)
        y_values.append(y)

    return _summarise({"x": x_values, "y": y_values}, spikes)


def test_seeded_corpus_has_analytic_schema_entries() -> None:
    """The seed corpus must expose deterministic analytic schema references."""
    names = list_reference_trace_specs()

    assert names == tuple(sorted(names))
    assert set(_DETERMINISTIC_SCHEMA_TRACES.values()) <= set(names)

    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    assert isinstance(spec, ReferenceTraceSpec)
    assert spec.schema_name == "lif"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.protocol.state_variables == ("v",)
    assert spec.protocol.inputs["I"] == 1.0


def test_reference_trace_corpus_covers_every_deterministic_bundled_schema() -> None:
    """Every deterministic bundled schema must have one committed trace."""
    deterministic_schemas = set(list_bundled_schemas()) - _STOCHASTIC_SCHEMA_NAMES

    assert set(_DETERMINISTIC_SCHEMA_TRACES) == deterministic_schemas
    for schema_name, trace_name in _DETERMINISTIC_SCHEMA_TRACES.items():
        spec = load_reference_trace_spec(trace_name)
        assert spec.schema_name == schema_name
        assert spec.runner == "universal_dsl"
        assert spec.provenance.source.endswith(f"/{schema_name}.toml")
        assert spec.provenance.citation is not None
        assert spec.provenance.citation
        if "doi" in trace_name:
            assert spec.provenance.citation.startswith("doi:")


def test_lif_seed_features_match_independent_closed_form_solution() -> None:
    """Committed LIF features must match the closed-form RC solution, not the runner."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")

    expected = _closed_form_features(
        initial=-65.0,
        steady=-55.0,
        tau=10.0,
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_quadratic_if_trace_features_match_independent_analytic_solution() -> None:
    """Committed QIF features must match the analytic zero-current Riccati flow."""
    spec = load_reference_trace_spec("quadratic_if_zero_current_analytic")

    expected = _quadratic_if_zero_current_features(
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "quadratic_if"
    assert spec.provenance.citation == "doi:10.1152/jn.2000.83.2.808"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_perfect_integrator_trace_features_match_independent_sawtooth_solution() -> None:
    """Committed perfect-integrator features must match the exact reset sawtooth."""
    spec = load_reference_trace_spec("perfect_integrator_constant_current_sawtooth")

    expected = _perfect_integrator_sawtooth_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "perfect_integrator"
    assert spec.provenance.kind == "analytic_sawtooth"
    assert spec.provenance.citation == "doi:10.1017/CBO9781107447615"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_theta_trace_features_match_independent_phase_solution() -> None:
    """Committed theta features must match the tangent half-angle phase solution."""
    spec = load_reference_trace_spec("theta_constant_current_phase_analytic")

    expected = _theta_constant_current_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "theta"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


_PARITY_CASES: list[tuple[str, str, str, str, Callable[[ReferenceTraceSpec], dict[str, float]]]] = [
    (
        "resonate_fire_subthreshold_resonance_doi",
        "resonate_fire",
        "analytic_linear_euler_reference",
        "doi:10.1162/089976601300014538",
        lambda spec: _resonate_fire_linear_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "glif_constant_current_threshold_adaptation",
        "glif",
        "analytic_linear_euler_reference",
        "doi:10.1038/s41467-017-02717-4",
        lambda spec: _glif_subthreshold_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "izhikevich_regular_spiking_doi",
        "izhikevich",
        "independent_euler_reference",
        "doi:10.1109/TNN.2003.820440",
        lambda spec: _izhikevich_rs_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "izhikevich2007_regular_spiking_doi",
        "izhikevich2007",
        "independent_euler_reference",
        "doi:10.7551/mitpress/2526.001.0001",
        lambda spec: _izhikevich2007_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "dpi_neuron_driven_spiking_doi",
        "dpi_neuron",
        "independent_euler_reference",
        "doi:10.1109/JPROC.2014.2313954",
        lambda spec: _dpi_neuron_driven_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "mihalas_niebur_driven_spiking_doi",
        "mihalas_niebur",
        "independent_rk4_reference",
        "doi:10.1162/neco.2008.12-07-680",
        lambda spec: _mihalas_niebur_driven_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "fitzhugh_nagumo_driven_oscillation_doi",
        "fitzhugh_nagumo",
        "independent_rk4_reference",
        "doi:10.1016/S0006-3495(61)86902-6",
        lambda spec: _fitzhugh_nagumo_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "fitzhugh_rinzel_driven_bursting_doi",
        "fitzhugh_rinzel",
        "independent_rk4_reference",
        "doi:10.1007/978-3-642-93360-8_26",
        lambda spec: _fitzhugh_rinzel_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "pernarowski_autonomous_bursting_doi",
        "pernarowski",
        "independent_rk4_reference",
        "doi:10.1137/S003613999223449X",
        lambda spec: _pernarowski_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "terman_wang_legion_oscillation_doi",
        "terman_wang",
        "independent_rk4_reference",
        "doi:10.1016/0167-2789(94)00205-5",
        lambda spec: _terman_wang_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "wilson_hr_driven_spiking_doi",
        "wilson_hr",
        "independent_rk4_reference",
        "doi:10.1006/jtbi.1999.1002",
        lambda spec: _wilson_hr_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "mckean_driven_oscillation_doi",
        "mckean",
        "independent_rk4_reference",
        "doi:10.1016/0001-8708(70)90023-X",
        lambda spec: _mckean_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "adex_resting_adaptation_doi",
        "adex",
        "independent_euler_reference",
        "doi:10.1152/jn.00686.2005",
        lambda spec: _adex_subthreshold_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "exp_if_resting_exponential_doi",
        "exp_if",
        "independent_euler_reference",
        "doi:10.1523/JNEUROSCI.23-37-11628.2003",
        lambda spec: _exp_if_subthreshold_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "hindmarsh_rose_short_bursting_prefix",
        "hindmarsh_rose",
        "independent_euler_reference",
        "doi:10.1098/rspb.1984.0024",
        lambda spec: _hindmarsh_rose_prefix_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "morris_lecar_driven_oscillation_doi",
        "morris_lecar",
        "independent_rk4_reference",
        "doi:10.1016/S0006-3495(81)84782-0",
        lambda spec: _morris_lecar_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "hodgkin_huxley_driven_spiking_doi",
        "hodgkin_huxley",
        "independent_macrostep_rk4_reference",
        "doi:10.1113/jphysiol.1952.sp004764",
        lambda spec: _hodgkin_huxley_macrostep_rk4_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=100,
        ),
    ),
    (
        "connor_stevens_driven_spiking_doi",
        "connor_stevens",
        "independent_macrostep_rk4_reference",
        "doi:10.1113/jphysiol.1971.sp009366",
        lambda spec: _connor_stevens_macrostep_rk4_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=100,
        ),
    ),
    (
        "wang_buzsaki_driven_spiking_doi",
        "wang_buzsaki",
        "independent_macrostep_gauss_seidel_reference",
        "doi:10.1523/JNEUROSCI.16-20-06402.1996",
        lambda spec: _wang_buzsaki_macrostep_gauss_seidel_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=50,
        ),
    ),
]


@pytest.mark.parametrize(
    ("trace_name", "schema_name", "kind", "citation", "reference"),
    _PARITY_CASES,
    ids=[case[1] for case in _PARITY_CASES],
)
def test_trace_features_match_independent_reference(
    trace_name: str,
    schema_name: str,
    kind: str,
    citation: str,
    reference: Callable[[ReferenceTraceSpec], dict[str, float]],
) -> None:
    """Each committed trace must reproduce an independent re-derivation to ``1e-12``.

    The per-case ``reference`` callable recomputes the expected feature map from the
    model's published equations (an explicit-Euler or analytic recurrence), so a
    passing assertion proves the committed corpus is independently reproduced rather
    than regenerated by the schema runner itself. The committed feature set must match
    the reference set exactly and every value to ``1e-12``.
    """
    spec = load_reference_trace_spec(trace_name)

    expected = reference(spec)

    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_rulkov_map_trace_features_match_independent_map_iteration() -> None:
    """Committed Rulkov features must match an independent piecewise-map iteration."""
    spec = load_reference_trace_spec("rulkov_map_driven_spiking_doi")

    expected = _rulkov_map_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "rulkov_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1103/PhysRevE.65.041922"
    assert spec.expected_features["spike_count"] > 0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_simulation_exercises_universal_schema_runner() -> None:
    """The harness must execute the committed schema through UniversalNeuron."""
    spec = load_reference_trace_spec("lapicque_constant_current_closed_form")

    simulation = simulate_reference_trace(spec)

    assert simulation.name == spec.name
    assert simulation.steps == spec.protocol.steps
    assert tuple(simulation.trace) == ("v",)
    assert len(simulation.trace["v"]) == spec.protocol.steps
    assert simulation.trace["v"][0] > 0.0
    assert simulation.spikes == tuple(0 for _ in range(spec.protocol.steps))
    assert simulation.features["spike_count"] == 0.0
    assert simulation.features["first_spike_step"] == -1.0
    assert simulation.features["max.v"] == max(simulation.trace["v"])


def test_reference_trace_validation_accepts_seeded_corpus() -> None:
    """All committed seed references must pass their own tolerance contracts."""
    reports = validate_all_reference_traces()

    assert {report.name for report in reports} == set(list_reference_trace_specs())
    assert all(report.passed for report in reports)
    assert all(report.mismatches == () for report in reports)


def test_name_based_simulation_and_validation_paths_are_public() -> None:
    """Name-based public helpers must route through the committed corpus."""
    simulation = simulate_reference_trace("lif_constant_current_closed_form")
    report = validate_reference_trace("lif_constant_current_closed_form")

    assert simulation.name == "lif_constant_current_closed_form"
    assert report.name == simulation.name
    assert report.passed


def test_unknown_reference_trace_name_fails_closed() -> None:
    """Unknown corpus identifiers must not silently fall back to another trace."""
    with pytest.raises(ValueError, match="unknown reference trace"):
        load_reference_trace_spec("not_a_committed_reference")


def test_validation_reports_feature_drift() -> None:
    """A drifted expected feature must fail closed with the feature name."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    drifted_features = dict(spec.expected_features)
    drifted_features["final.v"] += 0.25
    drifted = replace(spec, expected_features=drifted_features)

    report = validate_reference_trace_spec(drifted)

    assert not report.passed
    assert [mismatch.feature for mismatch in report.mismatches] == ["final.v"]


def test_simulation_rejects_unsupported_in_memory_runner() -> None:
    """In-memory specs cannot select runners outside the v1 production surface."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    unsupported = replace(spec, runner="python_loop")

    with pytest.raises(ValueError, match="unsupported reference-trace runner"):
        simulate_reference_trace(unsupported)
