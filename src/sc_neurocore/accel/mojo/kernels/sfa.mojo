# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration contract for SFA RK4

# The Python, Rust, Go, and Julia implementations are stateful. This Mojo
# surface keeps the same candidate-first RK4 equations as stateless helpers for
# accelerator integration and benchmark parity.

fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn sfa_valid(
    v: Float64,
    g_sfa: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_sfa: Float64,
    delta_g: Float64,
    e_k: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and v >= -200.0
        and v <= 100.0
        and _finite(g_sfa)
        and g_sfa >= 0.0
        and g_sfa <= 1.0e9
        and _finite(v_rest)
        and _finite(v_reset)
        and v_reset >= -200.0
        and v_reset <= 100.0
        and _finite(v_threshold)
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(tau_sfa)
        and tau_sfa > 0.0
        and _finite(delta_g)
        and delta_g >= 0.0
        and delta_g <= 1.0e9
        and _finite(e_k)
        and _finite(resistance)
        and resistance > 0.0
        and _finite(dt)
        and dt > 0.0
    )


fn sfa_derivative_v(
    v: Float64,
    g_sfa: Float64,
    current: Float64,
    v_rest: Float64,
    tau_m: Float64,
    e_k: Float64,
    resistance: Float64,
) -> Float64:
    return (-(v - v_rest) - g_sfa * (v - e_k) + resistance * current) / tau_m


fn sfa_derivative_g(g_sfa: Float64, tau_sfa: Float64) -> Float64:
    return -g_sfa / tau_sfa


fn sfa_next_v(
    v: Float64,
    g_sfa: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_sfa: Float64,
    delta_g: Float64,
    e_k: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not sfa_valid(v, g_sfa, v_rest, v_reset, v_threshold, tau_m, tau_sfa, delta_g, e_k, resistance, dt):
        return 0.0 / 0.0
    var k1v = sfa_derivative_v(v, g_sfa, current, v_rest, tau_m, e_k, resistance)
    var k1g = sfa_derivative_g(g_sfa, tau_sfa)
    var k2v = sfa_derivative_v(v + 0.5 * dt * k1v, g_sfa + 0.5 * dt * k1g, current, v_rest, tau_m, e_k, resistance)
    var k2g = sfa_derivative_g(g_sfa + 0.5 * dt * k1g, tau_sfa)
    var k3v = sfa_derivative_v(v + 0.5 * dt * k2v, g_sfa + 0.5 * dt * k2g, current, v_rest, tau_m, e_k, resistance)
    var k3g = sfa_derivative_g(g_sfa + 0.5 * dt * k2g, tau_sfa)
    var k4v = sfa_derivative_v(v + dt * k3v, g_sfa + dt * k3g, current, v_rest, tau_m, e_k, resistance)
    return v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


fn sfa_next_g(
    v: Float64,
    g_sfa: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_sfa: Float64,
    delta_g: Float64,
    e_k: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not sfa_valid(v, g_sfa, v_rest, v_reset, v_threshold, tau_m, tau_sfa, delta_g, e_k, resistance, dt):
        return 0.0 / 0.0
    var k1g = sfa_derivative_g(g_sfa, tau_sfa)
    var k2g = sfa_derivative_g(g_sfa + 0.5 * dt * k1g, tau_sfa)
    var k3g = sfa_derivative_g(g_sfa + 0.5 * dt * k2g, tau_sfa)
    var k4g = sfa_derivative_g(g_sfa + dt * k3g, tau_sfa)
    return g_sfa + (dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)


fn sfa_step_spike(
    v: Float64,
    g_sfa: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_sfa: Float64,
    delta_g: Float64,
    e_k: Float64,
    resistance: Float64,
    dt: Float64,
) -> Int:
    var next_v = sfa_next_v(v, g_sfa, current, v_rest, v_reset, v_threshold, tau_m, tau_sfa, delta_g, e_k, resistance, dt)
    var next_g = sfa_next_g(v, g_sfa, current, v_rest, v_reset, v_threshold, tau_m, tau_sfa, delta_g, e_k, resistance, dt)
    if not (_finite(next_v) and _finite(next_g)):
        return -1
    if not (next_v >= -200.0 and next_v <= 100.0 and next_g >= 0.0 and next_g <= 1.0e9):
        return -1
    if next_v >= v_threshold:
        var after_spike = next_g + delta_g
        if not (_finite(after_spike) and after_spike <= 1.0e9):
            return -1
        return 1
    return 0
