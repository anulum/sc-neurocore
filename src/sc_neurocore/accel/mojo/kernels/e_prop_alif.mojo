# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for e_prop_alif

fn valid(v: Float64, a: Float64, e_trace: Float64, tau_m: Float64, tau_a: Float64, v_threshold_base: Float64, beta: Float64, v_reset: Float64, dt: Float64, alpha_m: Float64, alpha_a: Float64) -> Bool:
    return v.is_finite() and a.is_finite() and a >= 0.0 and e_trace.is_finite() and tau_m.is_finite() and tau_m > 0.0 and tau_a.is_finite() and tau_a > 0.0 and v_threshold_base.is_finite() and beta.is_finite() and beta >= 0.0 and v_reset.is_finite() and dt.is_finite() and dt > 0.0 and dt <= tau_m and dt <= tau_a and v_threshold_base > v_reset and alpha_m.is_finite() and alpha_m > 0.0 and alpha_m < 1.0 and alpha_a.is_finite() and alpha_a > 0.0 and alpha_a < 1.0

fn step(v: Float64, a: Float64, e_trace: Float64, tau_m: Float64, tau_a: Float64, v_threshold_base: Float64, beta: Float64, v_reset: Float64, dt: Float64, alpha_m: Float64, alpha_a: Float64, current: Float64) -> Int:
    if not valid(v, a, e_trace, tau_m, tau_a, v_threshold_base, beta, v_reset, dt, alpha_m, alpha_a) or not current.is_finite():
        return 0
    var next_v = alpha_m * v + current
    var threshold = v_threshold_base + beta * a
    if next_v >= threshold:
        return 1
    return 0

fn reset() -> Int:
    return 0
