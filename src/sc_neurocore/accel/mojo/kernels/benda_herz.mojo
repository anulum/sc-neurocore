# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for benda_herz

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn benda_herz_valid(
    a: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
    tau_a: Float64,
    delta_a: Float64,
    dt: Float64,
    rng_threshold: Float64,
) -> Bool:
    return (
        _finite(a)
        and a >= 0.0
        and _finite(f_max)
        and f_max > 0.0
        and _finite(beta)
        and beta > 0.0
        and _finite(i_half)
        and _finite(tau_a)
        and tau_a > 0.0
        and _finite(delta_a)
        and delta_a >= 0.0
        and _finite(dt)
        and dt > 0.0
        and _finite(rng_threshold)
        and rng_threshold >= 0.0
        and rng_threshold < 1.0
    )


fn benda_herz_f_onset(
    x: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
) -> Float64:
    var z = beta * (x - i_half)
    if z > 1.7976931348623157e308:
        return f_max
    if z < -1.7976931348623157e308:
        return 0.0
    if z >= 0.0:
        return f_max / (1.0 + exp(-z))
    var exp_z = exp(z)
    return f_max * exp_z / (1.0 + exp_z)


fn benda_herz_rhs(
    a: Float64,
    current: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
    tau_a: Float64,
    delta_a: Float64,
) -> Float64:
    var rate = benda_herz_f_onset(current - a, f_max, beta, i_half)
    return -a / tau_a + delta_a * rate


fn benda_herz_next_a(
    a: Float64,
    current: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
    tau_a: Float64,
    delta_a: Float64,
    dt: Float64,
) -> Float64:
    var k1 = benda_herz_rhs(a, current, f_max, beta, i_half, tau_a, delta_a)
    var k2 = benda_herz_rhs(
        a + 0.5 * dt * k1, current, f_max, beta, i_half, tau_a, delta_a
    )
    var k3 = benda_herz_rhs(
        a + 0.5 * dt * k2, current, f_max, beta, i_half, tau_a, delta_a
    )
    var k4 = benda_herz_rhs(
        a + dt * k3, current, f_max, beta, i_half, tau_a, delta_a
    )
    return a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


fn benda_herz_probability(
    a: Float64,
    current: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
    tau_a: Float64,
    delta_a: Float64,
    dt: Float64,
) -> Float64:
    var r1 = benda_herz_f_onset(current - a, f_max, beta, i_half)
    var k1 = -a / tau_a + delta_a * r1
    var a2 = a + 0.5 * dt * k1
    var r2 = benda_herz_f_onset(current - a2, f_max, beta, i_half)
    var k2 = -a2 / tau_a + delta_a * r2
    var a3 = a + 0.5 * dt * k2
    var r3 = benda_herz_f_onset(current - a3, f_max, beta, i_half)
    var k3 = -a3 / tau_a + delta_a * r3
    var a4 = a + dt * k3
    var r4 = benda_herz_f_onset(current - a4, f_max, beta, i_half)
    var average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    return 1.0 - exp(-(average_rate * dt / 1000.0))


fn benda_herz_step_spike(
    a: Float64,
    current: Float64,
    f_max: Float64,
    beta: Float64,
    i_half: Float64,
    tau_a: Float64,
    delta_a: Float64,
    dt: Float64,
    rng_threshold: Float64,
) -> Int:
    if not _finite(current):
        return 0
    if not benda_herz_valid(
        a, f_max, beta, i_half, tau_a, delta_a, dt, rng_threshold
    ):
        return 0

    var next_a = benda_herz_next_a(a, current, f_max, beta, i_half, tau_a, delta_a, dt)
    var p = benda_herz_probability(a, current, f_max, beta, i_half, tau_a, delta_a, dt)
    if not _finite(next_a) or next_a < 0.0:
        return 0
    if not _finite(p) or p < 0.0 or p > 1.0:
        return 0
    if rng_threshold < p:
        return 1
    return 0
