# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for adaptive_threshold_if

fn valid(v: Float64, theta: Float64, v_rest: Float64, v_reset: Float64, theta_rest: Float64, delta_theta: Float64, tau_m: Float64, tau_theta: Float64, dt: Float64) -> Bool:
    return v.is_finite() and theta.is_finite() and v_rest.is_finite() and v_reset.is_finite() and theta_rest.is_finite() and delta_theta.is_finite() and delta_theta >= 0.0 and tau_m.is_finite() and tau_m > 0.0 and tau_theta.is_finite() and tau_theta > 0.0 and dt.is_finite() and dt > 0.0 and dt <= tau_m and dt <= tau_theta and theta_rest > v_rest and theta_rest > v_reset

fn step(v: Float64, theta: Float64, v_rest: Float64, v_reset: Float64, theta_rest: Float64, delta_theta: Float64, tau_m: Float64, tau_theta: Float64, dt: Float64, current: Float64) -> Int:
    if not valid(v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt) or not current.is_finite():
        return 0
    var next_v = v + (-(v - v_rest) + current) / tau_m * dt
    var next_theta = theta + (-(theta - theta_rest)) / tau_theta * dt
    if next_v >= next_theta:
        return 1
    return 0

fn reset() -> Int:
    return 0
