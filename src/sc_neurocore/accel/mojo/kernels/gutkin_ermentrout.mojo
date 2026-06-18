# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for gutkin_ermentrout

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn gutkin_ermentrout_valid(
    v: Float64,
    n: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_l: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(n)
        and n >= 0.0
        and n <= 1.0
        and _finite(g_na)
        and g_na >= 0.0
        and _finite(g_k)
        and g_k >= 0.0
        and _finite(g_l)
        and g_l >= 0.0
        and _finite(e_na)
        and _finite(e_k)
        and _finite(e_l)
        and _finite(dt)
        and dt > 0.0
        and _finite(v_threshold)
    )


fn _m_inf(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-(v + 20.0) / 15.0))


fn _n_inf(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-(v + 25.0) / 5.0))


fn _rhs_v(
    v: Float64,
    n: Float64,
    current: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_l: Float64,
) -> Float64:
    var i_na = g_na * _m_inf(v) * (v - e_na)
    var i_k = g_k * n * (v - e_k)
    var i_l = g_l * (v - e_l)
    return -i_na - i_k - i_l + current


fn _rhs_n(v: Float64, n: Float64) -> Float64:
    return _n_inf(v) - n


fn gutkin_ermentrout_next_v(
    v: Float64,
    n: Float64,
    current: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_l: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Float64:
    if not _finite(current) or not gutkin_ermentrout_valid(
        v, n, g_na, g_k, g_l, e_na, e_k, e_l, dt, v_threshold
    ):
        return 0.0 / 0.0
    var k1_v = _rhs_v(v, n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k1_n = _rhs_n(v, n)
    var k2_v = _rhs_v(v + 0.5 * dt * k1_v, n + 0.5 * dt * k1_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k2_n = _rhs_n(v + 0.5 * dt * k1_v, n + 0.5 * dt * k1_n)
    var k3_v = _rhs_v(v + 0.5 * dt * k2_v, n + 0.5 * dt * k2_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k3_n = _rhs_n(v + 0.5 * dt * k2_v, n + 0.5 * dt * k2_n)
    var k4_v = _rhs_v(v + dt * k3_v, n + dt * k3_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var next_v = v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    if not _finite(next_v):
        return 0.0 / 0.0
    return next_v


fn gutkin_ermentrout_next_n(
    v: Float64,
    n: Float64,
    current: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_l: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Float64:
    if not _finite(current) or not gutkin_ermentrout_valid(
        v, n, g_na, g_k, g_l, e_na, e_k, e_l, dt, v_threshold
    ):
        return 0.0 / 0.0
    var k1_v = _rhs_v(v, n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k1_n = _rhs_n(v, n)
    var k2_v = _rhs_v(v + 0.5 * dt * k1_v, n + 0.5 * dt * k1_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k2_n = _rhs_n(v + 0.5 * dt * k1_v, n + 0.5 * dt * k1_n)
    var k3_v = _rhs_v(v + 0.5 * dt * k2_v, n + 0.5 * dt * k2_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k3_n = _rhs_n(v + 0.5 * dt * k2_v, n + 0.5 * dt * k2_n)
    var k4_v = _rhs_v(v + dt * k3_v, n + dt * k3_n, current, g_na, g_k, g_l, e_na, e_k, e_l)
    var k4_n = _rhs_n(v + dt * k3_v, n + dt * k3_n)
    var next_n = n + dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0
    if not _finite(next_n) or next_n < 0.0 or next_n > 1.0:
        return 0.0 / 0.0
    return next_n


fn gutkin_ermentrout_step_spike(
    v: Float64,
    next_v: Float64,
    v_threshold: Float64,
) -> Int:
    if not _finite(v) or not _finite(next_v) or not _finite(v_threshold):
        return -1
    if next_v >= v_threshold and v < v_threshold:
        return 1
    return 0
