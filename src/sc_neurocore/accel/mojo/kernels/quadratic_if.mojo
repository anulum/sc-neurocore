# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for quadratic_if


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn quadratic_if_valid(
    v: Float64, v_reset: Float64, v_peak: Float64, dt: Float64
) -> Bool:
    return (
        _finite(v)
        and _finite(v_reset)
        and _finite(v_peak)
        and _finite(dt)
        and v < v_peak
        and v_reset < v_peak
        and dt > 0.0
    )


fn quadratic_if_step_spike(
    v: Float64, current: Float64, v_reset: Float64, v_peak: Float64, dt: Float64
) -> Int:
    if not _finite(current):
        return 0
    if not quadratic_if_valid(v, v_reset, v_peak, dt):
        return 0

    var increment = (v * v + current) * dt
    var next_v = v + increment
    if not _finite(increment) or not _finite(next_v):
        return 0
    if next_v >= v_peak:
        return 1
    return 0
