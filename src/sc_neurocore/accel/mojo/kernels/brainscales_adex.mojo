# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for brainscales_adex

from std.math import exp


fn _brainscales_adex_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn _brainscales_adex_clip(value: Float64, lower: Float64, upper: Float64) -> Float64:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


fn brainscales_adex_valid(
    v: Float64,
    w: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    delta_t: Float64,
    v_rh: Float64,
    tau: Float64,
    tau_w: Float64,
    a: Float64,
    b: Float64,
    hw_speedup: Float64,
    dt: Float64,
) -> Bool:
    return (
        _brainscales_adex_finite(v)
        and _brainscales_adex_finite(w)
        and _brainscales_adex_finite(v_rest)
        and _brainscales_adex_finite(v_reset)
        and _brainscales_adex_finite(v_threshold)
        and _brainscales_adex_finite(delta_t)
        and _brainscales_adex_finite(v_rh)
        and _brainscales_adex_finite(tau)
        and _brainscales_adex_finite(tau_w)
        and _brainscales_adex_finite(a)
        and _brainscales_adex_finite(b)
        and _brainscales_adex_finite(hw_speedup)
        and _brainscales_adex_finite(dt)
        and delta_t > 0.0
        and tau > 0.0
        and tau_w > 0.0
        and hw_speedup > 0.0
        and dt > 0.0
    )


fn brainscales_adex_step_spike(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    delta_t: Float64,
    v_rh: Float64,
    tau: Float64,
    tau_w: Float64,
    a: Float64,
    b: Float64,
    hw_speedup: Float64,
    dt: Float64,
) -> Int:
    if not _brainscales_adex_finite(current) or not brainscales_adex_valid(v, w, v_rest, v_reset, v_threshold, delta_t, v_rh, tau, tau_w, a, b, hw_speedup, dt):
        return -1

    var dt_hw = dt * hw_speedup
    var dt_bio = dt_hw / hw_speedup
    var exp_arg = _brainscales_adex_clip((v - v_rh) / delta_t, -20.0, 20.0)
    var exp_term = delta_t * exp(exp_arg)
    var dv = (-(v - v_rest) + exp_term - w + current) / tau * dt_bio
    var dw = (a * (v - v_rest) - w) / tau_w * dt_bio
    var next_v = v + dv
    var next_w = w + dw
    if not _brainscales_adex_finite(dt_hw) or not _brainscales_adex_finite(dt_bio) or not _brainscales_adex_finite(exp_term) or not _brainscales_adex_finite(dv) or not _brainscales_adex_finite(dw) or not _brainscales_adex_finite(next_v) or not _brainscales_adex_finite(next_w):
        return -1
    if next_v >= v_threshold:
        var spike_w = next_w + b
        if not _brainscales_adex_finite(spike_w):
            return -1
        return 1
    return 0


struct BrainScaleSAdExKernel:
    fn step(self, current: Float64) -> Int:
        return brainscales_adex_step_spike(
            -65.0,
            0.0,
            current,
            -65.0,
            -68.0,
            -50.0,
            2.0,
            -55.0,
            20.0,
            100.0,
            0.5,
            7.0,
            1000.0,
            0.1,
        )
