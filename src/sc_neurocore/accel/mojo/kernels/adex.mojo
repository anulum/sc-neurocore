# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for adex

from std.math import exp


fn _adex_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn _adex_clip(value: Float64, lower: Float64, upper: Float64) -> Float64:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


fn adex_valid(
    v: Float64,
    w: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    tau_w: Float64,
    a: Float64,
    b: Float64,
    c_m: Float64,
    dt: Float64,
) -> Bool:
    return (
        _adex_finite(v)
        and _adex_finite(w)
        and _adex_finite(v_rest)
        and _adex_finite(v_reset)
        and _adex_finite(v_threshold)
        and _adex_finite(v_rh)
        and _adex_finite(delta_t)
        and _adex_finite(tau)
        and _adex_finite(tau_w)
        and _adex_finite(a)
        and _adex_finite(b)
        and _adex_finite(c_m)
        and _adex_finite(dt)
        and delta_t > 0.0
        and tau > 0.0
        and tau_w > 0.0
        and c_m > 0.0
        and dt > 0.0
    )


fn adex_step_spike(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    tau_w: Float64,
    a: Float64,
    b: Float64,
    c_m: Float64,
    dt: Float64,
) -> Int:
    if not _adex_finite(current) or not adex_valid(v, w, v_rest, v_reset, v_threshold, v_rh, delta_t, tau, tau_w, a, b, c_m, dt):
        return -1

    var arg = _adex_clip((v - v_rh) / delta_t, -20.0, 20.0)
    var exp_term = delta_t * exp(arg)
    var dv = ((-(v - v_rest) + exp_term) / tau + (-w + current) / c_m) * dt
    var dw = (a * (v - v_rest) - w) / tau_w * dt
    var next_v = v + dv
    var next_w = w + dw
    if not _adex_finite(exp_term) or not _adex_finite(dv) or not _adex_finite(dw) or not _adex_finite(next_v) or not _adex_finite(next_w):
        return -1
    if next_v >= v_threshold:
        var spike_w = next_w + b
        if not _adex_finite(spike_w):
            return -1
        return 1
    return 0


struct AdExKernel:
    fn step(self, current: Float64) -> Int:
        return adex_step_spike(
            -65.0,
            0.0,
            current,
            -65.0,
            -68.0,
            -50.0,
            -55.0,
            2.0,
            20.0,
            100.0,
            0.5,
            7.0,
            200.0,
            0.1,
        )
