# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for adaptive_threshold_moe


fn _adaptive_threshold_moe_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn adaptive_threshold_moe_valid(
    k: Float64,
    ema_alpha: Float64,
    v: Float64,
    v_th: Float64,
    mean_abs_x: Float64,
) -> Bool:
    return (
        _adaptive_threshold_moe_finite(k)
        and k > 0.0
        and _adaptive_threshold_moe_finite(ema_alpha)
        and ema_alpha > 0.0
        and ema_alpha <= 1.0
        and _adaptive_threshold_moe_finite(v)
        and _adaptive_threshold_moe_finite(v_th)
        and v_th > 0.0
        and _adaptive_threshold_moe_finite(mean_abs_x)
        and mean_abs_x >= 0.0
    )


fn _adaptive_threshold_moe_round(value: Float64) -> Int:
    var lower = Int(value)
    var fraction = value - Float64(lower)
    if fraction > 0.5:
        return lower + 1
    if fraction < 0.5:
        return lower
    if lower % 2 == 0:
        return lower
    return lower + 1


fn adaptive_threshold_moe_step(
    k: Float64,
    ema_alpha: Float64,
    v: Float64,
    v_th: Float64,
    mean_abs_x: Float64,
    current: Float64,
) -> Int:
    if not adaptive_threshold_moe_valid(k, ema_alpha, v, v_th, mean_abs_x) or not _adaptive_threshold_moe_finite(current):
        return -1
    var magnitude = current
    if magnitude < 0.0:
        magnitude = -magnitude
    var next_mean_abs_x = (1.0 - ema_alpha) * mean_abs_x + ema_alpha * magnitude
    var next_v_th = 1.0
    if next_mean_abs_x > 1e-12:
        next_v_th = next_mean_abs_x / k
    if not _adaptive_threshold_moe_finite(next_v_th) or next_v_th <= 0.0:
        return -1
    var next_v = v + current
    if not _adaptive_threshold_moe_finite(next_v):
        return -1
    var ratio = next_v / next_v_th
    if not _adaptive_threshold_moe_finite(ratio):
        return -1
    var spikes = _adaptive_threshold_moe_round(ratio)
    if spikes < 0:
        spikes = 0
    var residual = next_v
    if spikes != 0:
        residual = next_v - next_v_th * Float64(spikes)
    if not _adaptive_threshold_moe_finite(residual):
        return -1
    return spikes
