# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for Rall cable


fn _finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


fn _solve5(
    alpha: Float64,
    g_ratio: Float64,
    v_rest: Float64,
    current: Float64,
    v0: Float64,
    v1: Float64,
    v2: Float64,
    v3: Float64,
    v4: Float64,
    index: Int,
) -> Float64:
    if not (
        _finite(alpha)
        and alpha > 0.0
        and _finite(g_ratio)
        and g_ratio >= 0.0
        and _finite(v_rest)
        and _finite(current)
        and _finite(v0)
        and _finite(v1)
        and _finite(v2)
        and _finite(v3)
        and _finite(v4)
    ):
        return 0.0 / 0.0

    var offdiag = -alpha * g_ratio
    var d0 = 1.0 + alpha + alpha * g_ratio
    var d1 = 1.0 + alpha + 2.0 * alpha * g_ratio
    var d2 = d1
    var d3 = d1
    var d4 = d0
    var r0 = v0 - v_rest
    var r1 = v1 - v_rest
    var r2 = v2 - v_rest
    var r3 = v3 - v_rest
    var r4 = v4 - v_rest + alpha * current

    if d0 == 0.0:
        return 0.0 / 0.0
    var c0 = offdiag / d0
    var y0 = r0 / d0

    var p1 = d1 - offdiag * c0
    if p1 == 0.0:
        return 0.0 / 0.0
    var c1 = offdiag / p1
    var y1 = (r1 - offdiag * y0) / p1

    var p2 = d2 - offdiag * c1
    if p2 == 0.0:
        return 0.0 / 0.0
    var c2 = offdiag / p2
    var y2 = (r2 - offdiag * y1) / p2

    var p3 = d3 - offdiag * c2
    if p3 == 0.0:
        return 0.0 / 0.0
    var c3 = offdiag / p3
    var y3 = (r3 - offdiag * y2) / p3

    var p4 = d4 - offdiag * c3
    if p4 == 0.0:
        return 0.0 / 0.0
    var x4 = (r4 - offdiag * y3) / p4
    var x3 = y3 - c3 * x4
    var x2 = y2 - c2 * x3
    var x1 = y1 - c1 * x2
    var x0 = y0 - c0 * x1

    if index == 0:
        return x0 + v_rest
    if index == 1:
        return x1 + v_rest
    if index == 2:
        return x2 + v_rest
    if index == 3:
        return x3 + v_rest
    if index == 4:
        return x4 + v_rest
    return 0.0 / 0.0


fn rall_cable_next5(
    tau_m: Float64,
    v_rest: Float64,
    g_ratio: Float64,
    dt: Float64,
    current: Float64,
    v0: Float64,
    v1: Float64,
    v2: Float64,
    v3: Float64,
    v4: Float64,
    index: Int,
) -> Float64:
    if not (_finite(tau_m) and tau_m > 0.0 and _finite(dt) and dt > 0.0):
        return 0.0 / 0.0
    return _solve5(dt / tau_m, g_ratio, v_rest, current, v0, v1, v2, v3, v4, index)


fn rall_cable_spike(next_soma: Float64, previous_soma: Float64, v_threshold: Float64) -> Int:
    if not (_finite(next_soma) and _finite(previous_soma) and _finite(v_threshold)):
        return -1
    if next_soma >= v_threshold and previous_soma < v_threshold:
        return 1
    return 0
