# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar contract for GLIF RK4 dynamics

# The active Python, Rust, Go, and Julia surfaces use a candidate-first RK4
# update for the GLIF four-state flow. This Mojo surface records the same
# scalar contract for later kernel specialisation.

fn finite(value: Float64) -> Bool:
    return value == value

fn derivative_v(v: Float64, v_rest: Float64, i_asc1: Float64, i_asc2: Float64, current: Float64, resistance: Float64, tau_m: Float64) -> Float64:
    return (-(v - v_rest) + resistance * current + i_asc1 + i_asc2) / tau_m

fn derivative_theta(v: Float64, theta: Float64, v_rest: Float64, theta_inf: Float64, a_theta: Float64, tau_theta: Float64) -> Float64:
    return (theta_inf - theta + a_theta * (v - v_rest)) / tau_theta

fn derivative_current(value: Float64, tau: Float64) -> Float64:
    return -value / tau

fn rk4_reference_voltage(v: Float64, v_rest: Float64, i_asc1: Float64, i_asc2: Float64, current: Float64, resistance: Float64, tau_m: Float64, dt: Float64) -> Float64:
    var k1 = derivative_v(v, v_rest, i_asc1, i_asc2, current, resistance, tau_m)
    var k2 = derivative_v(v + 0.5 * dt * k1, v_rest, i_asc1, i_asc2, current, resistance, tau_m)
    var k3 = derivative_v(v + 0.5 * dt * k2, v_rest, i_asc1, i_asc2, current, resistance, tau_m)
    var k4 = derivative_v(v + dt * k3, v_rest, i_asc1, i_asc2, current, resistance, tau_m)
    return v + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
