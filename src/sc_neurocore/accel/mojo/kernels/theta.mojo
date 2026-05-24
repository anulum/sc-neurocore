# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for theta

from std.math import cos

comptime PI = 3.14159265358979323846


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn theta_valid(theta: Float64, dt: Float64) -> Bool:
    return _finite(theta) and _finite(dt) and dt > 0.0


fn theta_step_spike(theta: Float64, current: Float64, dt: Float64) -> Int:
    if not _finite(current):
        return 0
    if not theta_valid(theta, dt):
        return 0

    var cos_theta = cos(theta)
    var dtheta = ((1.0 - cos_theta) + (1.0 + cos_theta) * current) * dt
    var next_theta = theta + dtheta
    if not _finite(dtheta) or not _finite(next_theta):
        return 0
    if theta < PI * 0.99 and next_theta >= PI * 0.99:
        return 1
    return 0
