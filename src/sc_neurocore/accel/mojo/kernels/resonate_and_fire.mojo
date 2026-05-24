# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for resonate_and_fire

from std.math import sqrt


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
        return 0
    if not resonate_and_fire_valid(x, y, b, omega, threshold, dt):
        return 0

    var dx = (b * x - omega * y + current) * dt
    var dy = (omega * x + b * y) * dt
    var next_x = x + dx
    var next_y = y + dy
    var radius_squared = next_x * next_x + next_y * next_y
    var radius = sqrt(radius_squared)
    if (
        not _finite(dx)
        or not _finite(dy)
        or not _finite(next_x)
        or not _finite(next_y)
        or not _finite(radius_squared)
        or not _finite(radius)
    ):
        return 0
    if radius >= threshold:
        return 1
    return 0
