# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration contract for COBA LIF RK4

# The Python, Rust, Go, and Julia implementations are stateful. This Mojo
# surface keeps the same candidate-first RK4 equations as stateless helpers for
# accelerator integration and benchmark parity.

fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn coba_lif_valid(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and v >= -200.0
        and v <= 100.0
        and _finite(g_e)
        and g_e >= 0.0
        and g_e <= 1.0e9
        and _finite(g_i)
        and g_i >= 0.0
        and g_i <= 1.0e9
        and _finite(c_m)
        and c_m > 0.0
        and _finite(g_l)
        and g_l >= 0.0
        and _finite(e_l)
        and _finite(e_e)
        and _finite(e_i)
        and _finite(tau_e)
        and tau_e > 0.0
        and _finite(tau_i)
        and tau_i > 0.0
        and _finite(v_threshold)
        and _finite(v_reset)
        and v_reset >= -200.0
        and v_reset <= 100.0
        and _finite(dt)
        and dt > 0.0
    )


fn coba_lif_derivative_v(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
) -> Float64:
    var i_syn = g_e * (v - e_e) + g_i * (v - e_i)
    return (-g_l * (v - e_l) - i_syn + current) / c_m


fn coba_lif_derivative_ge(g_e: Float64, tau_e: Float64) -> Float64:
    return -g_e / tau_e


fn coba_lif_derivative_gi(g_i: Float64, tau_i: Float64) -> Float64:
    return -g_i / tau_i


fn coba_lif_next_v(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not coba_lif_valid(v, g_e, g_i, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt):
        return 0.0 / 0.0
    var k1v = coba_lif_derivative_v(v, g_e, g_i, current, c_m, g_l, e_l, e_e, e_i)
    var k1e = coba_lif_derivative_ge(g_e, tau_e)
    var k1i = coba_lif_derivative_gi(g_i, tau_i)
    var k2v = coba_lif_derivative_v(v + 0.5 * dt * k1v, g_e + 0.5 * dt * k1e, g_i + 0.5 * dt * k1i, current, c_m, g_l, e_l, e_e, e_i)
    var k2e = coba_lif_derivative_ge(g_e + 0.5 * dt * k1e, tau_e)
    var k2i = coba_lif_derivative_gi(g_i + 0.5 * dt * k1i, tau_i)
    var k3v = coba_lif_derivative_v(v + 0.5 * dt * k2v, g_e + 0.5 * dt * k2e, g_i + 0.5 * dt * k2i, current, c_m, g_l, e_l, e_e, e_i)
    var k3e = coba_lif_derivative_ge(g_e + 0.5 * dt * k2e, tau_e)
    var k3i = coba_lif_derivative_gi(g_i + 0.5 * dt * k2i, tau_i)
    var k4v = coba_lif_derivative_v(v + dt * k3v, g_e + dt * k3e, g_i + dt * k3i, current, c_m, g_l, e_l, e_e, e_i)
    return v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


fn coba_lif_next_ge(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not coba_lif_valid(v, g_e, g_i, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt):
        return 0.0 / 0.0
    var k1e = coba_lif_derivative_ge(g_e, tau_e)
    var k2e = coba_lif_derivative_ge(g_e + 0.5 * dt * k1e, tau_e)
    var k3e = coba_lif_derivative_ge(g_e + 0.5 * dt * k2e, tau_e)
    var k4e = coba_lif_derivative_ge(g_e + dt * k3e, tau_e)
    return g_e + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)


fn coba_lif_next_gi(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not coba_lif_valid(v, g_e, g_i, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt):
        return 0.0 / 0.0
    var k1i = coba_lif_derivative_gi(g_i, tau_i)
    var k2i = coba_lif_derivative_gi(g_i + 0.5 * dt * k1i, tau_i)
    var k3i = coba_lif_derivative_gi(g_i + 0.5 * dt * k2i, tau_i)
    var k4i = coba_lif_derivative_gi(g_i + dt * k3i, tau_i)
    return g_i + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)


fn coba_lif_step_spike(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Int:
    var next_v = coba_lif_next_v(v, g_e, g_i, current, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt)
    var next_ge = coba_lif_next_ge(v, g_e, g_i, current, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt)
    var next_gi = coba_lif_next_gi(v, g_e, g_i, current, c_m, g_l, e_l, e_e, e_i, tau_e, tau_i, v_threshold, v_reset, dt)
    if not (_finite(next_v) and _finite(next_ge) and _finite(next_gi)):
        return -1
    if not (next_v >= -200.0 and next_v <= 100.0 and next_ge >= 0.0 and next_gi >= 0.0):
        return -1
    if next_v >= v_threshold:
        return 1
    return 0
