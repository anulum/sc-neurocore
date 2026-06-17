# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for escape_rate

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn escape_rate_valid(
    v: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(v_rest)
        and _finite(v_reset)
        and _finite(v_threshold)
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(rho_0)
        and rho_0 > 0.0
        and _finite(delta_u)
        and delta_u > 0.0
        and _finite(resistance)
        and resistance > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn _safe_exp(x: Float64) -> Float64:
    if x > 700.0:
        return exp(700.0)
    if x < -700.0:
        return exp(-700.0)
    return exp(x)


fn escape_rate_next_v(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not escape_rate_valid(
        v, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, dt
    ):
        return 0.0 / 0.0
    var v_inf = v_rest + resistance * current
    var decay = exp(-dt / tau_m)
    var next_v = v_inf + (v - v_inf) * decay
    if not _finite(v_inf) or not _finite(decay) or not _finite(next_v):
        return 0.0 / 0.0
    return next_v


fn escape_rate_probability(
    voltage: Float64,
    v_threshold: Float64,
    rho_0: Float64,
    delta_u: Float64,
    dt: Float64,
) -> Float64:
    if (
        not _finite(voltage)
        or not _finite(v_threshold)
        or not _finite(rho_0)
        or rho_0 <= 0.0
        or not _finite(delta_u)
        or delta_u <= 0.0
        or not _finite(dt)
        or dt <= 0.0
    ):
        return 0.0 / 0.0
    var rate = rho_0 * _safe_exp((voltage - v_threshold) / delta_u)
    var hazard = rate * dt
    if not _finite(hazard) or hazard < 0.0:
        return 0.0 / 0.0
    var p = 1.0 - exp(-hazard)
    if not _finite(p) or p < 0.0 or p > 1.0:
        return 0.0 / 0.0
    return p


fn escape_rate_step_spike(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
    rng_threshold: Float64,
) -> Int:
    if not _finite(rng_threshold) or rng_threshold < 0.0 or rng_threshold >= 1.0:
        return -1
    var next_v = escape_rate_next_v(
        v, current, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, dt
    )
    var p = escape_rate_probability(next_v, v_threshold, rho_0, delta_u, dt)
    if not _finite(next_v) or not _finite(p):
        return -1
    if rng_threshold < p:
        return 1
    return 0
