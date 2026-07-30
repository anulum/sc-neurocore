# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo source MAT* acceleration contract

# Non-resetting MAT* helpers. Voltage uses forward Euler; threshold-history
# terms use exact exponential decay. The caller owns state and must commit all
# outputs from the same pre-step tuple. Units are mV, ms, nA, and megaohms.

from std.math import exp


def _mat_finite(value: Float64) -> Bool:
    """Return true for finite binary64 values."""
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


def mat_valid(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Bool:
    """Validate the complete source MAT* state and configuration."""
    return (
        _mat_finite(v)
        and v >= -200.0
        and v <= 200.0
        and _mat_finite(theta1)
        and theta1 >= 0.0
        and theta1 <= 1.0e9
        and _mat_finite(theta2)
        and theta2 >= 0.0
        and theta2 <= 1.0e9
        and _mat_finite(refractory_remaining)
        and refractory_remaining >= 0.0
        and _mat_finite(omega)
        and omega >= -1.0e9
        and omega <= 1.0e9
        and _mat_finite(tau_m)
        and tau_m > 0.0
        and _mat_finite(tau_1)
        and tau_1 > 0.0
        and _mat_finite(tau_2)
        and tau_2 > 0.0
        and _mat_finite(alpha_1)
        and alpha_1 >= 0.0
        and alpha_1 <= 1.0e9
        and _mat_finite(alpha_2)
        and alpha_2 >= 0.0
        and alpha_2 <= 1.0e9
        and _mat_finite(resistance)
        and resistance > 0.0
        and _mat_finite(refractory_period)
        and refractory_period >= 0.0
        and refractory_remaining <= refractory_period
        and _mat_finite(dt)
        and dt > 0.0
    )


def mat_candidate_v(
    v: Float64, current: Float64, tau_m: Float64, resistance: Float64, dt: Float64
) -> Float64:
    """Return the paper's forward-Euler membrane candidate."""
    return v + dt * (-v + resistance * current) / tau_m


def mat_candidate_theta(theta: Float64, tau: Float64, dt: Float64) -> Float64:
    """Return exact exponential threshold-history decay."""
    return theta * exp(-dt / tau)


def mat_candidate_refractory(refractory_remaining: Float64, dt: Float64) -> Float64:
    """Return the nonnegative refractory countdown candidate."""
    var candidate = refractory_remaining - dt
    if candidate < 0.0:
        return 0.0
    return candidate


def mat_step_spike(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Int:
    """Return `1` for an event, `0` for silence, or `-1` on invalid state."""
    if not _mat_finite(current):
        return -1
    if not mat_valid(v, theta1, theta2, refractory_remaining, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt):
        return -1
    var next_v = mat_candidate_v(v, current, tau_m, resistance, dt)
    var next_theta1 = mat_candidate_theta(theta1, tau_1, dt)
    var next_theta2 = mat_candidate_theta(theta2, tau_2, dt)
    var next_refractory = mat_candidate_refractory(refractory_remaining, dt)
    if not (_mat_finite(next_v) and _mat_finite(next_theta1) and _mat_finite(next_theta2) and _mat_finite(next_refractory)):
        return -1
    if not (next_v >= -200.0 and next_v <= 200.0 and next_theta1 >= 0.0 and next_theta1 <= 1.0e9 and next_theta2 >= 0.0 and next_theta2 <= 1.0e9):
        return -1
    if next_refractory == 0.0 and next_v >= omega + next_theta1 + next_theta2:
        if next_theta1 + alpha_1 > 1.0e9 or next_theta2 + alpha_2 > 1.0e9:
            return -1
        return 1
    return 0


def mat_next_v(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the non-resetting membrane output, or NaN on invalid input."""
    if mat_step_spike(v, theta1, theta2, refractory_remaining, current, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt) < 0:
        return 0.0 / 0.0
    return mat_candidate_v(v, current, tau_m, resistance, dt)


def mat_next_theta1(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step fast threshold-history state."""
    var spike = mat_step_spike(v, theta1, theta2, refractory_remaining, current, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt)
    if spike < 0:
        return 0.0 / 0.0
    return mat_candidate_theta(theta1, tau_1, dt) + Float64(spike) * alpha_1


def mat_next_theta2(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step slow threshold-history state."""
    var spike = mat_step_spike(v, theta1, theta2, refractory_remaining, current, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt)
    if spike < 0:
        return 0.0 / 0.0
    return mat_candidate_theta(theta2, tau_2, dt) + Float64(spike) * alpha_2


def mat_next_refractory(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    refractory_remaining: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step absolute-refractory state."""
    var spike = mat_step_spike(v, theta1, theta2, refractory_remaining, current, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt)
    if spike < 0:
        return 0.0 / 0.0
    if spike == 1:
        return refractory_period
    return mat_candidate_refractory(refractory_remaining, dt)
