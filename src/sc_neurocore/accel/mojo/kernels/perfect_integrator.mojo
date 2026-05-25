# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for perfect_integrator


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn perfect_integrator_valid(
    v: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(c_m)
        and c_m > 0.0
        and _finite(v_threshold)
        and _finite(v_reset)
        and v_threshold > v_reset
        and v < v_threshold
        and _finite(dt)
        and dt > 0.0
    )


fn perfect_integrator_step_spike(
    v: Float64,
    current: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return -1
    if not perfect_integrator_valid(v, c_m, v_threshold, v_reset, dt):
        return -1

    var voltage_increment = current / c_m * dt
    var next_v = v + voltage_increment
    if not _finite(voltage_increment) or not _finite(next_v):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0
