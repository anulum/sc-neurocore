# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar helpers for leaky_compete_fire

from std.math import exp


fn _lcf_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


fn leaky_compete_fire_valid(
    voltage: Float64,
    current: Float64,
    tau: Float64,
    threshold: Float64,
    w_inh: Float64,
    dt: Float64,
) -> Bool:
    return (
        _lcf_finite(voltage)
        and _lcf_finite(current)
        and _lcf_finite(tau)
        and _lcf_finite(threshold)
        and _lcf_finite(w_inh)
        and _lcf_finite(dt)
        and tau > 0.0
        and w_inh >= 0.0
        and dt > 0.0
    )


fn leaky_compete_fire_exact_voltage(
    voltage: Float64, current: Float64, tau: Float64, dt: Float64
) -> Float64:
    if not (
        _lcf_finite(voltage)
        and _lcf_finite(current)
        and _lcf_finite(tau)
        and tau > 0.0
        and _lcf_finite(dt)
        and dt > 0.0
    ):
        return -1.0
    var next_voltage = current + (voltage - current) * exp(-dt / tau)
    if not _lcf_finite(next_voltage):
        return -1.0
    return next_voltage


fn leaky_compete_fire_step_spike(
    voltage: Float64,
    current: Float64,
    tau: Float64,
    threshold: Float64,
    w_inh: Float64,
    dt: Float64,
) -> Int:
    if not leaky_compete_fire_valid(
        voltage, current, tau, threshold, w_inh, dt
    ):
        return -1
    var next_voltage = leaky_compete_fire_exact_voltage(
        voltage, current, tau, dt
    )
    if next_voltage < 0.0:
        return -1
    if next_voltage >= threshold:
        return 1
    return 0


fn reset() -> Int:
    return 0
