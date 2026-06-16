# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for theta

from std.math import atan, cos, exp, sqrt, tan

comptime PI = 3.14159265358979323846


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn theta_valid(theta: Float64, dt: Float64) -> Bool:
    return _finite(theta) and _finite(dt) and dt > 0.0


fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn theta_step_spike(theta: Float64, current: Float64, dt: Float64) -> Int:
    if not _finite(current):
        return -1
    if not theta_valid(theta, dt):
        return -1

    var y = tan(theta / 2.0)
    if current > 0.0:
        var root_i = sqrt(current)
        var phase = atan(y / root_i)
        var next_phase = phase + root_i * dt
        var next_y = root_i * tan(next_phase)
        if not _finite(next_y):
            return -1
        if next_phase >= PI / 2.0:
            return 1
        return 0
    if current == 0.0:
        var denominator = 1.0 - y * dt
        if denominator <= 0.0:
            return 1
        return 0

    var root_i_neg = sqrt(-current)
    if _abs(y + root_i_neg) <= 1.0e-15:
        return 0
    var ratio = (y - root_i_neg) / (y + root_i_neg)
    var evolved = ratio * exp(2.0 * root_i_neg * dt)
    var crossing_denominator = 1.0 - evolved
    if not _finite(evolved) or not _finite(crossing_denominator):
        return -1
    if (ratio < 1.0 and evolved >= 1.0) or _abs(crossing_denominator) <= 1.0e-15:
        return 1
    return 0
