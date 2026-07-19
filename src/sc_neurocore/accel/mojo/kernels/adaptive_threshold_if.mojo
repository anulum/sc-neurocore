# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar kernel for composite reduced adaptive-threshold IF

from std.math import exp


fn _finite(value: Float64) -> Bool:
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


fn adaptive_threshold_if_valid(
    v: Float64,
    theta: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(theta)
        and _finite(v_rest)
        and _finite(v_reset)
        and _finite(theta_rest)
        and theta_rest > v_rest
        and theta_rest > v_reset
        and _finite(delta_theta)
        and delta_theta >= 0.0
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(tau_theta)
        and tau_theta > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn adaptive_threshold_if_exact_relaxation(
    state: Float64,
    steady_state: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    return steady_state + (state - steady_state) * exp(-dt / tau)


fn adaptive_threshold_if_step_spike(
    v: Float64,
    theta: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return -1
    if not adaptive_threshold_if_valid(
        v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt
    ):
        return -1
    var next_v = adaptive_threshold_if_exact_relaxation(v, v_rest + current, tau_m, dt)
    var next_theta = adaptive_threshold_if_exact_relaxation(theta, theta_rest, tau_theta, dt)
    if not _finite(next_v) or not _finite(next_theta):
        return -1
    if next_v >= next_theta:
        if not _finite(next_theta + delta_theta):
            return -1
        return 1
    return 0
