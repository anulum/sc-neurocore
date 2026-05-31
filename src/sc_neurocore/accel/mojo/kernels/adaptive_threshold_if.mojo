# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for adaptive_threshold_if

from std.math import exp

fn _adaptive_threshold_if_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn valid(v: Float64, theta: Float64, v_rest: Float64, v_reset: Float64, theta_rest: Float64, delta_theta: Float64, tau_m: Float64, tau_theta: Float64, dt: Float64) -> Bool:
    return _adaptive_threshold_if_finite(v) and _adaptive_threshold_if_finite(theta) and _adaptive_threshold_if_finite(v_rest) and _adaptive_threshold_if_finite(v_reset) and _adaptive_threshold_if_finite(theta_rest) and _adaptive_threshold_if_finite(delta_theta) and delta_theta >= 0.0 and _adaptive_threshold_if_finite(tau_m) and tau_m > 0.0 and _adaptive_threshold_if_finite(tau_theta) and tau_theta > 0.0 and _adaptive_threshold_if_finite(dt) and dt > 0.0 and theta_rest > v_rest and theta_rest > v_reset

fn step(v: Float64, theta: Float64, v_rest: Float64, v_reset: Float64, theta_rest: Float64, delta_theta: Float64, tau_m: Float64, tau_theta: Float64, dt: Float64, current: Float64) -> Int:
    if not valid(v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt) or not _adaptive_threshold_if_finite(current):
        return -1
    var v_inf = v_rest + current
    var next_v = v_inf + (v - v_inf) * exp(-dt / tau_m)
    var next_theta = theta_rest + (theta - theta_rest) * exp(-dt / tau_theta)
    if not _adaptive_threshold_if_finite(next_v) or not _adaptive_threshold_if_finite(next_theta):
        return -1
    if next_v >= next_theta:
        var spike_theta = next_theta + delta_theta
        if not _adaptive_threshold_if_finite(spike_theta):
            return -1
        return 1
    return 0

fn reset() -> Int:
    return 0
