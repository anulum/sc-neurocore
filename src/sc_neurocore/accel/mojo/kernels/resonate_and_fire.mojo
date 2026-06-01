# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for resonate_and_fire

from std.math import cos, exp, sin, sqrt


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn resonate_and_fire_valid(
    x: Float64,
    y: Float64,
    b: Float64,
    omega: Float64,
    threshold: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(x)
        and _finite(y)
        and _finite(b)
        and _finite(omega)
        and omega > 0.0
        and _finite(threshold)
        and threshold > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn resonate_and_fire_exact_x(
    x: Float64,
    y: Float64,
    current: Float64,
    b: Float64,
    omega: Float64,
    dt: Float64,
) -> Float64:
    var denominator = b * b + omega * omega
    if not _finite(denominator) or denominator <= 0.0:
        return 0.0 / 0.0
    var x_ss = -b * current / denominator
    var y_ss = omega * current / denominator
    var decay = exp(b * dt)
    var angle = omega * dt
    var cos_angle = cos(angle)
    var sin_angle = sin(angle)
    if (
        not _finite(x_ss)
        or not _finite(y_ss)
        or not _finite(decay)
        or not _finite(angle)
        or not _finite(cos_angle)
        or not _finite(sin_angle)
    ):
        return 0.0 / 0.0
    var dx = x - x_ss
    var dy = y - y_ss
    return x_ss + decay * (dx * cos_angle - dy * sin_angle)


fn resonate_and_fire_exact_y(
    x: Float64,
    y: Float64,
    current: Float64,
    b: Float64,
    omega: Float64,
    dt: Float64,
) -> Float64:
    var denominator = b * b + omega * omega
    if not _finite(denominator) or denominator <= 0.0:
        return 0.0 / 0.0
    var x_ss = -b * current / denominator
    var y_ss = omega * current / denominator
    var decay = exp(b * dt)
    var angle = omega * dt
    var cos_angle = cos(angle)
    var sin_angle = sin(angle)
    if (
        not _finite(x_ss)
        or not _finite(y_ss)
        or not _finite(decay)
        or not _finite(angle)
        or not _finite(cos_angle)
        or not _finite(sin_angle)
    ):
        return 0.0 / 0.0
    var dx = x - x_ss
    var dy = y - y_ss
    return y_ss + decay * (dx * sin_angle + dy * cos_angle)


fn resonate_and_fire_step_spike(
    x: Float64,
    y: Float64,
    current: Float64,
    b: Float64,
    omega: Float64,
    threshold: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return -1
    if not resonate_and_fire_valid(x, y, b, omega, threshold, dt):
        return -1

    var next_x = resonate_and_fire_exact_x(x, y, current, b, omega, dt)
    var next_y = resonate_and_fire_exact_y(x, y, current, b, omega, dt)
    var radius_squared = next_x * next_x + next_y * next_y
    var radius = sqrt(radius_squared)
    if (
        not _finite(next_x)
        or not _finite(next_y)
        or not _finite(radius_squared)
        or not _finite(radius)
    ):
        return -1
    if radius >= threshold:
        return 1
    return 0
