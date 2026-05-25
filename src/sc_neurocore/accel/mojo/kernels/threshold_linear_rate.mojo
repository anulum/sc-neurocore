# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for threshold_linear_rate


fn _threshold_linear_rate_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn threshold_linear_rate_valid(r: Float64, theta: Float64, gain: Float64) -> Bool:
    return (
        _threshold_linear_rate_finite(r)
        and r >= 0.0
        and _threshold_linear_rate_finite(theta)
        and _threshold_linear_rate_finite(gain)
        and gain >= 0.0
    )


fn threshold_linear_rate_step(r: Float64, theta: Float64, gain: Float64, current: Float64) -> Float64:
    if not threshold_linear_rate_valid(r, theta, gain) or not _threshold_linear_rate_finite(current):
        return -1.0
    var drive = current - theta
    if drive < 0.0:
        drive = 0.0
    var next_r = gain * drive
    if not _threshold_linear_rate_finite(next_r) or next_r < 0.0:
        return -1.0
    return next_r


fn reset() -> Float64:
    return 0.0
