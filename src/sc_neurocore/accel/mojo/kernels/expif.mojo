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

    var arg = _expif_clip((v - v_rh) / delta_t, -20.0, 20.0)
    var exp_term = delta_t * exp(arg)
    var dv = (-(v - v_rest) + exp_term + current) / tau * dt
    var next_v = v + dv
    if not _expif_finite(exp_term) or not _expif_finite(dv) or not _expif_finite(next_v):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0


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
