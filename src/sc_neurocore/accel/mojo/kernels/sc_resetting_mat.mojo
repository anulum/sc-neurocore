# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration contract for SC resetting MAT

# The stateful Python, Rust, Go, and Julia paths preserve the same SC resetting MAT ODE with
# candidate-first RK4. This Mojo surface exposes stateless helpers so accelerator
# kernels can share the numerical contract without owning host-side state.

def _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


def sc_resetting_mat_valid(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and v >= -200.0
        and v <= 100.0
        and _finite(theta1)
        and theta1 >= 0.0
        and theta1 <= 1.0e9
        and _finite(theta2)
        and theta2 >= 0.0
        and theta2 <= 1.0e9
        and _finite(v_rest)
        and _finite(v_reset)
        and v_reset >= -200.0
        and v_reset <= 100.0
        and _finite(v_threshold_base)
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(tau_1)
        and tau_1 > 0.0
        and _finite(tau_2)
        and tau_2 > 0.0
        and _finite(h1)
        and h1 >= 0.0
        and h1 <= 1.0e9
        and _finite(h2)
        and h2 >= 0.0
        and h2 <= 1.0e9
        and _finite(resistance)
        and resistance > 0.0
        and _finite(dt)
        and dt > 0.0
    )


def sc_resetting_mat_derivative_v(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    tau_m: Float64,
    resistance: Float64,
) -> Float64:
    return (-(v - v_rest) + resistance * current) / tau_m


def sc_resetting_mat_derivative_theta(theta: Float64, tau: Float64) -> Float64:
    return -theta / tau


def sc_resetting_mat_next_v(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not sc_resetting_mat_valid(v, theta1, theta2, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt):
        return 0.0 / 0.0
    var k1v = sc_resetting_mat_derivative_v(v, current, v_rest, tau_m, resistance)
    var k2v = sc_resetting_mat_derivative_v(v + 0.5 * dt * k1v, current, v_rest, tau_m, resistance)
    var k3v = sc_resetting_mat_derivative_v(v + 0.5 * dt * k2v, current, v_rest, tau_m, resistance)
    var k4v = sc_resetting_mat_derivative_v(v + dt * k3v, current, v_rest, tau_m, resistance)
    return v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


def sc_resetting_mat_next_theta1(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not sc_resetting_mat_valid(v, theta1, theta2, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt):
        return 0.0 / 0.0
    var k1 = sc_resetting_mat_derivative_theta(theta1, tau_1)
    var k2 = sc_resetting_mat_derivative_theta(theta1 + 0.5 * dt * k1, tau_1)
    var k3 = sc_resetting_mat_derivative_theta(theta1 + 0.5 * dt * k2, tau_1)
    var k4 = sc_resetting_mat_derivative_theta(theta1 + dt * k3, tau_1)
    return theta1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sc_resetting_mat_next_theta2(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not sc_resetting_mat_valid(v, theta1, theta2, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt):
        return 0.0 / 0.0
    var k1 = sc_resetting_mat_derivative_theta(theta2, tau_2)
    var k2 = sc_resetting_mat_derivative_theta(theta2 + 0.5 * dt * k1, tau_2)
    var k3 = sc_resetting_mat_derivative_theta(theta2 + 0.5 * dt * k2, tau_2)
    var k4 = sc_resetting_mat_derivative_theta(theta2 + dt * k3, tau_2)
    return theta2 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sc_resetting_mat_step_spike(
    v: Float64,
    theta1: Float64,
    theta2: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
) -> Int:
    var next_v = sc_resetting_mat_next_v(v, theta1, theta2, current, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt)
    var next_theta1 = sc_resetting_mat_next_theta1(v, theta1, theta2, current, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt)
    var next_theta2 = sc_resetting_mat_next_theta2(v, theta1, theta2, current, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt)
    if not (_finite(next_v) and _finite(next_theta1) and _finite(next_theta2)):
        return -1
    if not (next_v >= -200.0 and next_v <= 100.0 and next_theta1 >= 0.0 and next_theta1 <= 1.0e9 and next_theta2 >= 0.0 and next_theta2 <= 1.0e9):
        return -1
    if next_v >= v_threshold_base + next_theta1 + next_theta2:
        var theta1_after_spike = next_theta1 + h1
        var theta2_after_spike = next_theta2 + h2
        if not (_finite(theta1_after_spike) and _finite(theta2_after_spike) and theta1_after_spike <= 1.0e9 and theta2_after_spike <= 1.0e9):
            return -1
        return 1
    return 0
