# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for spike_response

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn spike_response_valid(
    v: Float64,
    v_threshold: Float64,
    tau_eta: Float64,
    tau_kappa: Float64,
    eta_reset: Float64,
    time_since_spike: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(v_threshold)
        and _finite(tau_eta)
        and tau_eta > 0.0
        and _finite(tau_kappa)
        and tau_kappa > 0.0
        and _finite(eta_reset)
        and _finite(time_since_spike)
        and time_since_spike >= 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn spike_response_eta(
    time_since_spike: Float64,
    eta_reset: Float64,
    tau_eta: Float64,
) -> Float64:
    if not (
        _finite(time_since_spike)
        and time_since_spike >= 0.0
        and _finite(eta_reset)
        and _finite(tau_eta)
        and tau_eta > 0.0
    ):
        return 0.0 / 0.0
    if time_since_spike < 100.0:
        return eta_reset * exp(-time_since_spike / tau_eta)
    return 0.0


fn spike_response_kappa(
    weighted_input: Float64,
    dt: Float64,
    tau_kappa: Float64,
) -> Float64:
    if not (_finite(weighted_input) and _finite(dt) and dt > 0.0 and _finite(tau_kappa) and tau_kappa > 0.0):
        return 0.0 / 0.0
    return weighted_input * (1.0 - exp(-dt / tau_kappa))


fn spike_response_next_v(
    weighted_input: Float64,
    time_since_spike: Float64,
    eta_reset: Float64,
    tau_eta: Float64,
    dt: Float64,
    tau_kappa: Float64,
) -> Float64:
    var eta = spike_response_eta(time_since_spike, eta_reset, tau_eta)
    var kappa = spike_response_kappa(weighted_input, dt, tau_kappa)
    var next_v = eta + kappa
    if not _finite(next_v):
        return 0.0 / 0.0
    return next_v


fn spike_response_spike(
    v: Float64,
    v_threshold: Float64,
) -> Int:
    if not (_finite(v) and _finite(v_threshold)):
        return -1
    if v >= v_threshold:
        return 1
    return 0
