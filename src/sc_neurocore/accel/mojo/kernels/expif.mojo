# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for expif

from std.math import exp


fn _expif_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn _expif_clip(value: Float64, lower: Float64, upper: Float64) -> Float64:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


fn expif_valid(
    v: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    dt: Float64,
) -> Bool:
    return (
        _expif_finite(v)
        and _expif_finite(v_rest)
        and _expif_finite(v_reset)
        and _expif_finite(v_threshold)
        and _expif_finite(v_rh)
        and _expif_finite(delta_t)
        and _expif_finite(tau)
        and _expif_finite(dt)
        and delta_t > 0.0
        and tau > 0.0
        and dt > 0.0
    )


fn expif_step_spike(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    dt: Float64,
) -> Int:
    if not _expif_finite(current) or not expif_valid(v, v_rest, v_reset, v_threshold, v_rh, delta_t, tau, dt):
        return -1

    var k1 = _expif_rhs(v, current, v_rest, v_rh, delta_t, tau)
    var k2 = _expif_rhs(v + 0.5 * dt * k1, current, v_rest, v_rh, delta_t, tau)
    var k3 = _expif_rhs(v + 0.5 * dt * k2, current, v_rest, v_rh, delta_t, tau)
    var k4 = _expif_rhs(v + dt * k3, current, v_rest, v_rh, delta_t, tau)
    var next_v = v + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not _expif_finite(k1) or not _expif_finite(k2) or not _expif_finite(k3) or not _expif_finite(k4) or not _expif_finite(next_v):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0


fn expif_next_v(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    if not _expif_finite(current) or not expif_valid(v, v_rest, v_reset, v_threshold, v_rh, delta_t, tau, dt):
        return 0.0 / 0.0

    var k1 = _expif_rhs(v, current, v_rest, v_rh, delta_t, tau)
    var k2 = _expif_rhs(v + 0.5 * dt * k1, current, v_rest, v_rh, delta_t, tau)
    var k3 = _expif_rhs(v + 0.5 * dt * k2, current, v_rest, v_rh, delta_t, tau)
    var k4 = _expif_rhs(v + dt * k3, current, v_rest, v_rh, delta_t, tau)
    var next_v = v + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not _expif_finite(k1) or not _expif_finite(k2) or not _expif_finite(k3) or not _expif_finite(k4) or not _expif_finite(next_v):
        return 0.0 / 0.0
    if next_v >= v_threshold:
        return v_reset
    return next_v


struct ExpIFKernel:
    fn step(self, current: Float64) -> Int:
        return expif_step_spike(
            -65.0,
            current,
            -65.0,
            -68.0,
            -50.0,
            -55.0,
            2.0,
            20.0,
            0.1,
        )


fn _expif_rhs(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
) -> Float64:
    var arg = _expif_clip((v - v_rh) / delta_t, -20.0, 20.0)
    var exp_term = delta_t * exp(arg)
    var rhs = (-(v - v_rest) + exp_term + current) / tau
    if not _expif_finite(exp_term) or not _expif_finite(rhs):
        return 0.0 / 0.0
    return rhs
