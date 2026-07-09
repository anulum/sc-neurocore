# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Nonlinear LIF kernel contract.
#
# The Python, Rust, Go, and Julia implementations are stateful. This Mojo
# surface keeps the same validation and RK4 equations as a stateless contract
# for accelerator integration.

fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )

fn nlif_valid(
    v: Float64,
    w: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(w)
        and _finite(v_rest)
        and _finite(v_crit)
        and _finite(v_threshold)
        and _finite(v_reset)
        and _finite(a)
        and _finite(b)
        and _finite(tau_w)
        and _finite(c_m)
        and _finite(dt)
        and v_rest < v_crit
        and v_crit < v_threshold
        and v_reset < v_threshold
        and a >= 0.0
        and b >= 0.0
        and tau_w > 0.0
        and c_m > 0.0
        and dt > 0.0
        and dt <= tau_w
    )

fn nlif_step_spike(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return -1
    if not nlif_valid(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt):
        return -1

    var next_v = nlif_next_v(v, w, current, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt)
    if not _finite(next_v):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0

fn nlif_derivative_v(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_crit: Float64,
    a: Float64,
    c_m: Float64,
) -> Float64:
    return (a * (v - v_rest) * (v - v_crit) - w + current) / c_m

fn nlif_derivative_w(
    v: Float64,
    w: Float64,
    v_rest: Float64,
    b: Float64,
    tau_w: Float64,
) -> Float64:
    return (b * (v - v_rest) - w) / tau_w

fn nlif_next_v(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not nlif_valid(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt):
        return 0.0 / 0.0
    var k1v = nlif_derivative_v(v, w, current, v_rest, v_crit, a, c_m)
    var k1w = nlif_derivative_w(v, w, v_rest, b, tau_w)
    var k2v = nlif_derivative_v(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current, v_rest, v_crit, a, c_m)
    var k2w = nlif_derivative_w(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, v_rest, b, tau_w)
    var k3v = nlif_derivative_v(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current, v_rest, v_crit, a, c_m)
    var k3w = nlif_derivative_w(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, v_rest, b, tau_w)
    var k4v = nlif_derivative_v(v + dt * k3v, w + dt * k3w, current, v_rest, v_crit, a, c_m)
    return v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

fn nlif_next_w(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not nlif_valid(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt):
        return 0.0 / 0.0
    var k1v = nlif_derivative_v(v, w, current, v_rest, v_crit, a, c_m)
    var k1w = nlif_derivative_w(v, w, v_rest, b, tau_w)
    var k2v = nlif_derivative_v(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current, v_rest, v_crit, a, c_m)
    var k2w = nlif_derivative_w(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, v_rest, b, tau_w)
    var k3v = nlif_derivative_v(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current, v_rest, v_crit, a, c_m)
    var k3w = nlif_derivative_w(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, v_rest, b, tau_w)
    var k4w = nlif_derivative_w(v + dt * k3v, w + dt * k3w, v_rest, b, tau_w)
    return w + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
