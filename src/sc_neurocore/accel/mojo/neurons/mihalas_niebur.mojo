# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo source-faithful Mihalas-Niebur batch kernel
#
# Build:
#   mojo build --emit shared-lib -o libmihalasniebur.so mihalas_niebur.mojo
#
# The kernel implements equations 2.1–2.2 of doi:10.1162/neco.2008.12-07-680.
# Rates are per millisecond and current-like quantities are divided by
# capacitance. The output buffer contains n voltages followed by final
# (v, theta, i1, i2).

from std.memory import UnsafePointer


@always_inline
def _dv(
    v: Float64,
    i1: Float64,
    i2: Float64,
    current: Float64,
    v_rest: Float64,
    leak_rate: Float64,
) -> Float64:
    return current + i1 + i2 - leak_rate * (v - v_rest)


@always_inline
def _dtheta(
    v: Float64,
    theta: Float64,
    v_rest: Float64,
    theta_inf: Float64,
    threshold_voltage_coupling: Float64,
    threshold_decay_rate: Float64,
) -> Float64:
    var voltage_term = threshold_voltage_coupling * (v - v_rest)
    var decay_term = threshold_decay_rate * (theta - theta_inf)
    return voltage_term - decay_term


@export
def mihalas_niebur_simulate_c(
    v0: Float64,
    theta0: Float64,
    i1_0: Float64,
    i2_0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_reset: Float64,
    theta_inf: Float64,
    leak_rate: Float64,
    threshold_voltage_coupling: Float64,
    threshold_decay_rate: Float64,
    current_decay_rate_1: Float64,
    current_decay_rate_2: Float64,
    current_retention_1: Float64,
    current_retention_2: Float64,
    current_jump_1: Float64,
    current_jump_2: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var theta = theta0
    var i1 = i1_0
    var i2 = i2_0
    var half_dt = 0.5 * dt
    var events: Int64 = 0
    for index in range(n_steps):
        var k1v = _dv(v, i1, i2, current, v_rest, leak_rate)
        var k1t = _dtheta(
            v, theta, v_rest, theta_inf,
            threshold_voltage_coupling, threshold_decay_rate,
        )
        var k1i1 = -current_decay_rate_1 * i1
        var k1i2 = -current_decay_rate_2 * i2

        var v2 = v + half_dt * k1v
        var theta2 = theta + half_dt * k1t
        var i1_2 = i1 + half_dt * k1i1
        var i2_2 = i2 + half_dt * k1i2
        var k2v = _dv(v2, i1_2, i2_2, current, v_rest, leak_rate)
        var k2t = _dtheta(
            v2, theta2, v_rest, theta_inf,
            threshold_voltage_coupling, threshold_decay_rate,
        )
        var k2i1 = -current_decay_rate_1 * i1_2
        var k2i2 = -current_decay_rate_2 * i2_2

        var v3 = v + half_dt * k2v
        var theta3 = theta + half_dt * k2t
        var i1_3 = i1 + half_dt * k2i1
        var i2_3 = i2 + half_dt * k2i2
        var k3v = _dv(v3, i1_3, i2_3, current, v_rest, leak_rate)
        var k3t = _dtheta(
            v3, theta3, v_rest, theta_inf,
            threshold_voltage_coupling, threshold_decay_rate,
        )
        var k3i1 = -current_decay_rate_1 * i1_3
        var k3i2 = -current_decay_rate_2 * i2_3

        var v4 = v + dt * k3v
        var theta4 = theta + dt * k3t
        var i1_4 = i1 + dt * k3i1
        var i2_4 = i2 + dt * k3i2
        var k4v = _dv(v4, i1_4, i2_4, current, v_rest, leak_rate)
        var k4t = _dtheta(
            v4, theta4, v_rest, theta_inf,
            threshold_voltage_coupling, threshold_decay_rate,
        )
        var k4i1 = -current_decay_rate_1 * i1_4
        var k4i2 = -current_decay_rate_2 * i2_4

        v += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        theta += dt * (k1t + 2.0 * k2t + 2.0 * k3t + k4t) / 6.0
        i1 += dt * (k1i1 + 2.0 * k2i1 + 2.0 * k3i1 + k4i1) / 6.0
        i2 += dt * (k1i2 + 2.0 * k2i2 + 2.0 * k3i2 + k4i2) / 6.0

        if v >= theta:
            i1 = current_retention_1 * i1 + current_jump_1
            i2 = current_retention_2 * i2 + current_jump_2
            v = v_reset
            if theta_reset > theta:
                theta = theta_reset
            events += 1
        trace[index] = v
    trace[n_steps] = v
    trace[n_steps + 1] = theta
    trace[n_steps + 2] = i1
    trace[n_steps + 3] = i2
    return events
