# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for srm0

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn srm0_valid(
    v: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_eta: Float64,
    eta_reset: Float64,
    resistance: Float64,
    dt: Float64,
    eta: Float64,
    t: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(v_rest)
        and _finite(v_threshold)
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(tau_eta)
        and tau_eta > 0.0
        and _finite(eta_reset)
        and eta_reset >= 0.0
        and _finite(resistance)
        and _finite(dt)
        and dt > 0.0
        and _finite(eta)
        and _finite(t)
    )


fn _eta_coupling_integral(
    tau_m: Float64,
    tau_eta: Float64,
    dt: Float64,
) -> Float64:
    var membrane_decay = exp(-dt / tau_m)
    var eta_decay = exp(-dt / tau_eta)
    var rate_delta = (1.0 / tau_m) - (1.0 / tau_eta)
    if rate_delta < 1.0e-14 and rate_delta > -1.0e-14:
        return dt * membrane_decay / tau_m
    return (eta_decay - membrane_decay) / (tau_m * rate_delta)


fn srm0_next_eta(
    eta: Float64,
    tau_eta: Float64,
    dt: Float64,
) -> Float64:
    if not (_finite(eta) and _finite(tau_eta) and tau_eta > 0.0 and _finite(dt) and dt > 0.0):
        return 0.0 / 0.0
    var next_eta = eta * exp(-dt / tau_eta)
    if not _finite(next_eta):
        return 0.0 / 0.0
    return next_eta


fn srm0_next_v(
    v: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_eta: Float64,
    eta_reset: Float64,
    resistance: Float64,
    dt: Float64,
    eta: Float64,
    t: Float64,
    current: Float64,
) -> Float64:
    if not _finite(current) or not srm0_valid(
        v, v_rest, v_threshold, tau_m, tau_eta, eta_reset, resistance, dt, eta, t
    ):
        return 0.0 / 0.0
    var membrane_decay = exp(-dt / tau_m)
    var steady = v_rest + resistance * current
    var next_v = steady + (v - steady) * membrane_decay + eta * _eta_coupling_integral(tau_m, tau_eta, dt)
    if not _finite(next_v):
        return 0.0 / 0.0
    return next_v


fn srm0_step_spike(
    next_v: Float64,
    v_threshold: Float64,
) -> Int:
    if not _finite(next_v) or not _finite(v_threshold):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0
