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
    "connor_stevens": "connor_stevens_resting_gate_doi",
    "dpi_neuron": "dpi_neuron_driven_spiking_doi",
    "exp_if": "exp_if_resting_exponential_doi",
    "fitzhugh_nagumo": "fitzhugh_nagumo_driven_oscillation_doi",
    "glif": "glif_constant_current_threshold_adaptation",
    "hindmarsh_rose": "hindmarsh_rose_short_bursting_prefix",
    "hodgkin_huxley": "hodgkin_huxley_resting_gate_doi",
    "izhikevich": "izhikevich_regular_spiking_doi",
    "izhikevich2007": "izhikevich2007_regular_spiking_doi",
    "lapicque": "lapicque_constant_current_closed_form",
    "lif": "lif_constant_current_closed_form",
    "morris_lecar": "morris_lecar_depolarizing_current_doi",
    "perfect_integrator": "perfect_integrator_constant_current_sawtooth",
    "quadratic_if": "quadratic_if_zero_current_analytic",
    "resonate_fire": "resonate_fire_subthreshold_resonance_doi",
    "rulkov_map": "rulkov_map_driven_spiking_doi",
    "theta": "theta_constant_current_phase_analytic",
    "wang_buzsaki": "wang_buzsaki_resting_interneuron_doi",
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


def _fitzhugh_nagumo_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the driven FitzHugh-Nagumo recurrence.

    The FitzHugh (1961) cubic membrane and linear recovery equations are advanced
    with the same simultaneous explicit-Euler update the schema runner applies, and
    the ``v = -1`` reset (recovery ``w`` left unchanged) fires whenever the
    post-update membrane crosses the ``v > 1`` threshold. The reference is an
    independent re-derivation of the committed relaxation-oscillation trace, not a
    copy of the runner.

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
    v = -1.0
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = v - v**3 / 3 - w + current
        dw = epsilon * (v + a - b * w)
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > 1.0:
            spikes.append(1)
            v_next = -1.0
        else:
            spikes.append(0)
        v, w = v_next, w_next
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


def _morris_lecar_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the Morris-Lecar recurrence.

    The Morris-Lecar (1981) sigmoidal calcium activation and potassium gating
    equations are advanced with the same simultaneous explicit-Euler update the
    schema runner applies. ``numpy.tanh`` and ``numpy.cosh`` match the runner's
    activation and rate functions bit-for-bit, and the verbatim expression order is
    preserved so the recurrence reproduces the runner exactly. The depolarizing
    current stays below the ``v > 0`` threshold, so the identity reset never fires
    and the reference is an independent re-derivation of the committed trajectory.

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
    phi = 0.0667
    v = -60.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (
            -g_ca * 0.5 * (1 + float(np.tanh((v - v1) / v2))) * (v - e_ca)
            - g_k * w * (v - e_k)
            - g_l * (v - e_l)
            + current
        ) / c_m
        dw = (
            phi
            * float(np.cosh((v - v3) / (2 * v4)))
            * (0.5 * (1 + float(np.tanh((v - v3) / v4))) - w)
        )
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > 0:
            spikes.append(1)
        else:
            spikes.append(0)
        v, w = v_next, w_next
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


def _hodgkin_huxley_resting_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact explicit-Euler features for the resting Hodgkin-Huxley recurrence.

    The Hodgkin-Huxley (1952) membrane and the sodium/potassium gating variables are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies, reusing :func:`_np_exp` and :func:`_reference_exprel` so the sharpened
    rate functions match the runner bit-for-bit. The verbatim expression order is
    preserved; the resting zero-current protocol never crosses the ``v > 0``
    threshold and the schema declares no reset, so the reference is an independent
    re-derivation of the committed gate-relaxation trajectory.

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
        Reference feature map for the ``v``, ``m``, ``h``, and ``n`` state variables
        plus spike-count and first-spike-step features.
    """
    capacitance = 1.0
    g_na = 120.0
    g_k = 36.0
    g_l = 0.3
    e_na = 50.0
    e_k = -77.0
    e_l = -54.4
    v = -65.0
    m = 0.05
    h = 0.6
    n = 0.32
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": []}
    spikes: list[int] = []
    for _ in range(steps):
        dv = (
            -g_na * m**3 * h * (v - e_na) - g_k * n**4 * (v - e_k) - g_l * (v - e_l) + current
        ) / capacitance
        dm = 1.0 / _reference_exprel(-(v + 40) / 10) * (1 - m) - 4 * _np_exp(-(v + 65) / 18) * m
        dh = 0.07 * _np_exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 35) / 10)) * h
        dn = 0.1 / _reference_exprel(-(v + 55) / 10) * (1 - n) - 0.125 * _np_exp(-(v + 65) / 80) * n
        v_next = v + dv * dt
        m_next = m + dm * dt
        h_next = h + dh * dt
        n_next = n + dn * dt
        spikes.append(1 if v_next > 0 else 0)
        v, m, h, n = v_next, m_next, h_next, n_next
        recorded["v"].append(v)
        recorded["m"].append(m)
        recorded["h"].append(h)
        recorded["n"].append(n)

    return _summarise(recorded, spikes)


def _connor_stevens_resting_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact explicit-Euler features for the resting Connor-Stevens recurrence.

    The Connor-Stevens (1971) membrane and its six gating variables (fast sodium
    ``m``/``h``, delayed-rectifier ``n``, and A-type ``a``/``b``) are advanced with
    the same simultaneous explicit-Euler update the schema runner applies, reusing
    :func:`_np_exp` and :func:`_reference_exprel` so the rate functions match the
    runner bit-for-bit. The verbatim expression order is preserved; the resting
    zero-current protocol never crosses the ``v > 0`` threshold and the schema
    declares no reset, so the reference is an independent re-derivation of the
    committed gate trajectory.

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
    v = -68.0
    m = 0.01
    h = 0.99
    n = 0.1
    a = 0.5
    b = 0.1
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": [], "a": [], "b": []}
    spikes: list[int] = []
    for _ in range(steps):
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
        v_next = v + dv * dt
        m_next = m + dm * dt
        h_next = h + dh * dt
        n_next = n + dn * dt
        a_next = a + da * dt
        b_next = b + db * dt
        spikes.append(1 if v_next > 0 else 0)
        v, m, h, n, a, b = v_next, m_next, h_next, n_next, a_next, b_next
        recorded["v"].append(v)
        recorded["m"].append(m)
        recorded["h"].append(h)
        recorded["n"].append(n)
        recorded["a"].append(a)
        recorded["b"].append(b)

    return _summarise(recorded, spikes)


def _wang_buzsaki_resting_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact explicit-Euler features for the resting Wang-Buzsaki recurrence.

    The Wang-Buzsaki (1996) fast-spiking interneuron membrane and its ``h``/``n``
    gating variables (sodium activation is instantaneous) are advanced with the same
    simultaneous explicit-Euler update the schema runner applies, reusing
    :func:`_np_exp` so the rate functions match the runner bit-for-bit. The verbatim
    expression order is preserved; the resting zero-current protocol never crosses
    the ``v > -10`` threshold and the schema declares no reset, so the reference is
    an independent re-derivation of the committed gate trajectory.

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
        Reference feature map for the ``v``, ``h``, and ``n`` state variables plus
        spike-count and first-spike-step features.
    """
    capacitance = 1.0
    g_na = 35.0
    g_k = 9.0
    g_l = 0.1
    e_na = 55.0
    e_k = -90.0
    e_l = -65.0
    phi = 5.0
    v = -65.0
    h = 0.6
    n = 0.32
    recorded: dict[str, list[float]] = {"v": [], "h": [], "n": []}
    spikes: list[int] = []
    for _ in range(steps):
        dv = (
            -g_na * (1 / (1 + _np_exp(-(v + 35) / 10))) ** 3 * h * (v - e_na)
            - g_k * n**4 * (v - e_k)
            - g_l * (v - e_l)
            + current
        ) / capacitance
        dh = phi * (
            0.07 * _np_exp(-(v + 58) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 28) / 10)) * h
        )
        dn = phi * (
            0.01 * (v + 34) / (1 - _np_exp(-(v + 34) / 10)) * (1 - n)
            - 0.125 * _np_exp(-(v + 44) / 80) * n
        )
        v_next = v + dv * dt
        h_next = h + dh * dt
        n_next = n + dn * dt
        spikes.append(1 if v_next > -10 else 0)
        v, h, n = v_next, h_next, n_next
        recorded["v"].append(v)
        recorded["h"].append(h)
        recorded["n"].append(n)

    return _summarise(recorded, spikes)


def _rulkov_map_features(*, current: float, steps: int) -> dict[str, float]:
    """Return exact features for the Rulkov 2002 piecewise map iteration.

    The Rulkov (2002) fast/slow model is a discrete map, so an independent
    implementation of its three-branch fast map (rational subthreshold, spike
    plateau, hard reset) and slow drift reproduces the runner exactly — a map has no
    integration error, so independent parity is exact ground truth. Level spike
    detection (post-update ``x > 0``) matches the schema runner.

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
        if x <= 0:
            x_next = alpha / (1.0 - x) + y + current
        elif x < alpha + y + current:
            x_next = alpha + y + current
        else:
            x_next = -1.0
        y_next = y - mu * (x + 1.0) + mu * sigma
        x, y = x_next, y_next
        spikes.append(1 if x > 0.0 else 0)
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
        "fitzhugh_nagumo_driven_oscillation_doi",
        "fitzhugh_nagumo",
        "independent_euler_reference",
        "doi:10.1016/S0006-3495(61)86902-6",
        lambda spec: _fitzhugh_nagumo_euler_features(
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
        "morris_lecar_depolarizing_current_doi",
        "morris_lecar",
        "independent_euler_reference",
        "doi:10.1016/S0006-3495(81)84782-0",
        lambda spec: _morris_lecar_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "hodgkin_huxley_resting_gate_doi",
        "hodgkin_huxley",
        "independent_euler_reference",
        "doi:10.1113/jphysiol.1952.sp004764",
        lambda spec: _hodgkin_huxley_resting_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "connor_stevens_resting_gate_doi",
        "connor_stevens",
        "independent_euler_reference",
        "doi:10.1113/jphysiol.1971.sp009368",
        lambda spec: _connor_stevens_resting_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "wang_buzsaki_resting_interneuron_doi",
        "wang_buzsaki",
        "independent_euler_reference",
        "doi:10.1523/JNEUROSCI.16-20-06402.1996",
        lambda spec: _wang_buzsaki_resting_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
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
