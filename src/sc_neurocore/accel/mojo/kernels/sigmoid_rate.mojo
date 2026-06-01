# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for sigmoid_rate

from std.math import exp


fn _sigmoid_rate_finite(x: Float64) -> Bool:
    var residual = x - x
    return x == x and residual == 0.0


fn sigmoid_rate_valid(
    r: Float64, tau: Float64, beta: Float64, theta: Float64, dt: Float64
) -> Bool:
    return (
        _sigmoid_rate_finite(r)
        and _sigmoid_rate_finite(tau)
        and _sigmoid_rate_finite(beta)
        and _sigmoid_rate_finite(theta)
        and _sigmoid_rate_finite(dt)
        and r >= 0.0
        and r <= 1.0
        and tau > 0.0
        and dt > 0.0
    )


fn _sigmoid_rate_transfer(
    beta: Float64, current: Float64, theta: Float64
) -> Float64:
    var z = beta * (current - theta)
    if z != z:
        return -1.0
    if z > 1.7976931348623157e308:
        return 1.0
    if z < -1.7976931348623157e308:
        return 0.0
    if not _sigmoid_rate_finite(z):
        return -1.0
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    var exp_z = exp(z)
    return exp_z / (1.0 + exp_z)


fn sigmoid_rate_step(
    r: Float64,
    current: Float64,
    tau: Float64,
    beta: Float64,
    theta: Float64,
    dt: Float64,
) -> Float64:
    if not _sigmoid_rate_finite(current):
        return -1.0
    if not sigmoid_rate_valid(r, tau, beta, theta, dt):
        return -1.0
    var sigma = _sigmoid_rate_transfer(beta, current, theta)
    if sigma < 0.0:
        return -1.0
    var decay = exp(-dt / tau)
    var next_r = decay * r + (1.0 - decay) * sigma
    if not _sigmoid_rate_finite(next_r) or next_r < 0.0 or next_r > 1.0:
        return -1.0
    return next_r
