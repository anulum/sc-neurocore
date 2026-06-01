# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar contract for Mihalas-Niebur RK4 dynamics

# The active Python, Rust, Go, and Julia surfaces use a candidate-first RK4
# update for the Mihalas-Niebur four-state flow. This Mojo surface records the
# same scalar contract for downstream kernel specialization.

fn finite(value: Float64) -> Bool:
    return value == value

fn derivative_v(v: Float64, v_rest: Float64, i1: Float64, i2: Float64, current: Float64, tau_v: Float64) -> Float64:
    return (-(v - v_rest) + i1 + i2 + current) / tau_v

fn derivative_theta(v: Float64, theta: Float64, v_rest: Float64, theta_inf: Float64, a: Float64, tau_theta: Float64) -> Float64:
    return (theta_inf - theta + a * (v - v_rest)) / tau_theta

fn derivative_i(value: Float64, tau: Float64) -> Float64:
    return -value / tau

fn rk4_reference_voltage(v: Float64, v_rest: Float64, i1: Float64, i2: Float64, current: Float64, tau_v: Float64, dt: Float64) -> Float64:
    var k1 = derivative_v(v, v_rest, i1, i2, current, tau_v)
    var k2 = derivative_v(v + 0.5 * dt * k1, v_rest, i1, i2, current, tau_v)
    var k3 = derivative_v(v + 0.5 * dt * k2, v_rest, i1, i2, current, tau_v)
    var k4 = derivative_v(v + dt * k3, v_rest, i1, i2, current, tau_v)
    return v + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
