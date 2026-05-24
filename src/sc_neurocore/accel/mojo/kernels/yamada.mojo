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

    var m_inf = _sigmoid((v + 30.0) / 9.5)
    var n_inf = _sigmoid((v + 30.0) / 10.0)
    var q_inf = _sigmoid((v + 50.0) / 10.0)
    var tau_n = 1.0 + 7.5 / (1.0 + exp((v + 40.0) / 12.0))
    var i_na = g_na * m_inf * m_inf * m_inf * (1.0 - n) * (v - e_na)
    var i_k = g_k * n * n * n * n * (v - e_k)
    var i_q = g_q * q * (v - e_q)
    var i_l = g_l * (v - e_l)
    var dv = (-i_na - i_k - i_q - i_l + current) * dt
    var dn = (n_inf - n) / tau_n * dt
    var dq = (q_inf - q) / tau_q * dt
    var next_v = v + dv
    var next_n = n + dn
    var next_q = q + dq
    if (
        not _finite(m_inf)
        or not _finite(n_inf)
        or not _finite(q_inf)
        or not _finite(tau_n)
        or not _finite(i_na)
        or not _finite(i_k)
        or not _finite(i_q)
        or not _finite(i_l)
        or not _finite(dv)
        or not _finite(dn)
        or not _finite(dq)
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
