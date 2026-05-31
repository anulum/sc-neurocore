# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for yamada

from std.math import exp


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        var z = exp(-x)
        return 1.0 / (1.0 + z)
    var z = exp(x)
    return z / (1.0 + z)


fn _tau_n(v: Float64) -> Float64:
    var x = (v + 40.0) / 12.0
    if not _finite(x):
        return 0.0 / 0.0
    if x > 709.0:
        return 1.0
    return 1.0 + 7.5 / (1.0 + exp(x))


fn yamada_valid(
    v: Float64,
    n: Float64,
    q: Float64,
    g_na: Float64,
    g_k: Float64,
    g_q: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_q: Float64,
    e_l: Float64,
    tau_q: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(n)
        and n >= 0.0
        and n <= 1.0
        and _finite(q)
        and q >= 0.0
        and q <= 1.0
        and _finite(g_na)
        and g_na >= 0.0
        and _finite(g_k)
        and g_k >= 0.0
        and _finite(g_q)
        and g_q >= 0.0
        and _finite(g_l)
        and g_l >= 0.0
        and _finite(e_na)
        and _finite(e_k)
        and _finite(e_q)
        and _finite(e_l)
        and _finite(tau_q)
        and tau_q > 0.0
        and _finite(dt)
        and dt > 0.0
        and _finite(v_threshold)
    )


fn yamada_step_spike(
    v: Float64,
    n: Float64,
    q: Float64,
    current: Float64,
    g_na: Float64,
    g_k: Float64,
    g_q: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_q: Float64,
    e_l: Float64,
    tau_q: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Int:
    if not _finite(current):
        return 0
    if not yamada_valid(
        v,
        n,
        q,
        g_na,
        g_k,
        g_q,
        g_l,
        e_na,
        e_k,
        e_q,
        e_l,
        tau_q,
        dt,
        v_threshold,
    ):
        return 0

    var m1 = _sigmoid((v + 30.0) / 9.5)
    var n1_inf = _sigmoid((v + 30.0) / 10.0)
    var q1_inf = _sigmoid((v + 50.0) / 10.0)
    var tau1 = _tau_n(v)
    var ina1 = g_na * m1 * m1 * m1 * (1.0 - n) * (v - e_na)
    var ik1 = g_k * n * n * n * n * (v - e_k)
    var iq1 = g_q * q * (v - e_q)
    var il1 = g_l * (v - e_l)
    var k1_v = -ina1 - ik1 - iq1 - il1 + current
    var k1_n = (n1_inf - n) / tau1
    var k1_q = (q1_inf - q) / tau_q

    var v2 = v + 0.5 * dt * k1_v
    var n2 = n + 0.5 * dt * k1_n
    var q2 = q + 0.5 * dt * k1_q
    if not _finite(v2) or not _finite(n2) or not _finite(q2) or n2 < 0.0 or n2 > 1.0 or q2 < 0.0 or q2 > 1.0:
        return 0
    var m2 = _sigmoid((v2 + 30.0) / 9.5)
    var n2_inf = _sigmoid((v2 + 30.0) / 10.0)
    var q2_inf = _sigmoid((v2 + 50.0) / 10.0)
    var tau2 = _tau_n(v2)
    var ina2 = g_na * m2 * m2 * m2 * (1.0 - n2) * (v2 - e_na)
    var ik2 = g_k * n2 * n2 * n2 * n2 * (v2 - e_k)
    var iq2 = g_q * q2 * (v2 - e_q)
    var il2 = g_l * (v2 - e_l)
    var k2_v = -ina2 - ik2 - iq2 - il2 + current
    var k2_n = (n2_inf - n2) / tau2
    var k2_q = (q2_inf - q2) / tau_q

    var v3 = v + 0.5 * dt * k2_v
    var n3 = n + 0.5 * dt * k2_n
    var q3 = q + 0.5 * dt * k2_q
    if not _finite(v3) or not _finite(n3) or not _finite(q3) or n3 < 0.0 or n3 > 1.0 or q3 < 0.0 or q3 > 1.0:
        return 0
    var m3 = _sigmoid((v3 + 30.0) / 9.5)
    var n3_inf = _sigmoid((v3 + 30.0) / 10.0)
    var q3_inf = _sigmoid((v3 + 50.0) / 10.0)
    var tau3 = _tau_n(v3)
    var ina3 = g_na * m3 * m3 * m3 * (1.0 - n3) * (v3 - e_na)
    var ik3 = g_k * n3 * n3 * n3 * n3 * (v3 - e_k)
    var iq3 = g_q * q3 * (v3 - e_q)
    var il3 = g_l * (v3 - e_l)
    var k3_v = -ina3 - ik3 - iq3 - il3 + current
    var k3_n = (n3_inf - n3) / tau3
    var k3_q = (q3_inf - q3) / tau_q

    var v4 = v + dt * k3_v
    var n4 = n + dt * k3_n
    var q4 = q + dt * k3_q
    if not _finite(v4) or not _finite(n4) or not _finite(q4) or n4 < 0.0 or n4 > 1.0 or q4 < 0.0 or q4 > 1.0:
        return 0
    var m4 = _sigmoid((v4 + 30.0) / 9.5)
    var n4_inf = _sigmoid((v4 + 30.0) / 10.0)
    var q4_inf = _sigmoid((v4 + 50.0) / 10.0)
    var tau4 = _tau_n(v4)
    var ina4 = g_na * m4 * m4 * m4 * (1.0 - n4) * (v4 - e_na)
    var ik4 = g_k * n4 * n4 * n4 * n4 * (v4 - e_k)
    var iq4 = g_q * q4 * (v4 - e_q)
    var il4 = g_l * (v4 - e_l)
    var k4_v = -ina4 - ik4 - iq4 - il4 + current
    var k4_n = (n4_inf - n4) / tau4
    var k4_q = (q4_inf - q4) / tau_q

    var next_v = v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    var next_n = n + dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0
    var next_q = q + dt * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q) / 6.0
    if (
        not _finite(k1_v)
        or not _finite(k1_n)
        or not _finite(k1_q)
        or not _finite(k2_v)
        or not _finite(k2_n)
        or not _finite(k2_q)
        or not _finite(k3_v)
        or not _finite(k3_n)
        or not _finite(k3_q)
        or not _finite(k4_v)
        or not _finite(k4_n)
        or not _finite(k4_q)
        or not _finite(next_v)
        or not _finite(next_n)
        or not _finite(next_q)
        or next_n < 0.0
        or next_n > 1.0
        or next_q < 0.0
        or next_q > 1.0
    ):
        return 0
    if next_v >= v_threshold and v < v_threshold:
        return 1
    return 0
