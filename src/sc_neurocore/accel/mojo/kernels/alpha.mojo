# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar kernel for dual alpha-synapse LIF

from std.math import abs, exp


fn _finite(value: Float64) -> Bool:
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


fn alpha_valid(
    v: Float64,
    a_exc: Float64,
    i_exc: Float64,
    a_inh: Float64,
    i_inh: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_v: Float64,
    tau_exc: Float64,
    tau_inh: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(a_exc)
        and _finite(i_exc)
        and _finite(a_inh)
        and _finite(i_inh)
        and _finite(v_rest)
        and _finite(v_threshold)
        and v_threshold > v_rest
        and _finite(tau_v)
        and tau_v > 0.0
        and _finite(tau_exc)
        and tau_exc > 0.0
        and _finite(tau_inh)
        and tau_inh > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn alpha_filter_next(
    rise_state: Float64,
    current_state: Float64,
    drive: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    var steady_state = tau * drive
    var rise_delta = rise_state - steady_state
    var current_delta = current_state - steady_state
    return steady_state + exp(-dt / tau) * (current_delta + rise_delta * dt / tau)


fn alpha_rise_next(
    rise_state: Float64,
    drive: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    var steady_state = tau * drive
    return steady_state + (rise_state - steady_state) * exp(-dt / tau)


fn alpha_drive_contribution(
    current_delta: Float64,
    rise_delta: Float64,
    tau_drive: Float64,
    tau_v: Float64,
    dt: Float64,
) -> Float64:
    var rate_v = 1.0 / tau_v
    var rate_drive = 1.0 / tau_drive
    var decay_v = exp(-dt / tau_v)
    var decay_drive = exp(-dt / tau_drive)
    if abs(rate_v - rate_drive) <= 1.0e-14:
        return rate_v * decay_v * (
            current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive)
        )
    var rate_delta = rate_v - rate_drive
    var first_order = current_delta * (decay_drive - decay_v) / rate_delta
    var second_order = rise_delta / tau_drive * (
        decay_drive * (rate_delta * dt - 1.0) + decay_v
    ) / (rate_delta * rate_delta)
    return rate_v * (first_order + second_order)


fn alpha_step_spike(
    v: Float64,
    a_exc: Float64,
    i_exc: Float64,
    a_inh: Float64,
    i_inh: Float64,
    exc_current: Float64,
    inh_current: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_v: Float64,
    tau_exc: Float64,
    tau_inh: Float64,
    dt: Float64,
) -> Int:
    if not _finite(exc_current) or not _finite(inh_current):
        return -1
    if not alpha_valid(
        v, a_exc, i_exc, a_inh, i_inh, v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt
    ):
        return -1
    var exc_steady = tau_exc * exc_current
    var inh_steady = tau_inh * inh_current
    var v_steady = v_rest + exc_steady - inh_steady
    var decay_v = exp(-dt / tau_v)
    var v_next = v_steady + (v - v_steady) * decay_v + alpha_drive_contribution(
        i_exc - exc_steady, a_exc - exc_steady, tau_exc, tau_v, dt
    ) - alpha_drive_contribution(
        i_inh - inh_steady, a_inh - inh_steady, tau_inh, tau_v, dt
    )
    if not _finite(v_next):
        return -1
    if v_next >= v_threshold:
        return 1
    return 0
