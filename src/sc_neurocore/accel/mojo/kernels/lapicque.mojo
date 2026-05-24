# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for lapicque


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn lapicque_valid(
    v: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(v_rest)
        and _finite(v_reset)
        and _finite(v_threshold)
        and v_threshold > v_rest
        and v_threshold > v_reset
        and v < v_threshold
        and _finite(tau)
        and tau > 0.0
        and _finite(resistance)
        and resistance > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn lapicque_step_spike(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return 0
    if not lapicque_valid(v, v_rest, v_reset, v_threshold, tau, resistance, dt):
        return 0

    var dv = (-(v - v_rest) + resistance * current) / tau * dt
    var next_v = v + dv
    if not _finite(dv) or not _finite(next_v):
        return 0
    if next_v >= v_threshold:
        return 1
    return 0
