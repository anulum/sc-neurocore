# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for morris_lecar

from std.math import cosh, tanh


fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn morris_lecar_valid(
    v: Float64,
    w: Float64,
    c_m: Float64,
    g_ca: Float64,
    g_k: Float64,
    g_l: Float64,
    e_ca: Float64,
    e_k: Float64,
    e_l: Float64,
    v1: Float64,
    v2: Float64,
    v3: Float64,
    v4: Float64,
    phi: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(w)
        and w >= 0.0
        and w <= 1.0
        and _finite(c_m)
        and c_m > 0.0
        and _finite(g_ca)
        and g_ca > 0.0
        and _finite(g_k)
        and g_k > 0.0
        and _finite(g_l)
        and g_l > 0.0
        and _finite(e_ca)
        and _finite(e_k)
        and _finite(e_l)
        and _finite(v1)
        and _finite(v2)
        and v2 > 0.0
        and _finite(v3)
        and _finite(v4)
        and v4 > 0.0
        and _finite(phi)
        and phi > 0.0
        and _finite(dt)
        and dt > 0.0
        and _finite(v_threshold)
    )


fn _m_inf(v: Float64, v1: Float64, v2: Float64) -> Float64:
    return 0.5 * (1.0 + tanh((v - v1) / v2))


fn _w_inf(v: Float64, v3: Float64, v4: Float64) -> Float64:
    return 0.5 * (1.0 + tanh((v - v3) / v4))


fn _lambda(v: Float64, v3: Float64, v4: Float64, phi: Float64) -> Float64:
    return phi * cosh((v - v3) / (2.0 * v4))


fn _rhs_v(
    v: Float64,
    w: Float64,
    current: Float64,
    c_m: Float64,
    g_ca: Float64,
    g_k: Float64,
    g_l: Float64,
    e_ca: Float64,
    e_k: Float64,
    e_l: Float64,
    v1: Float64,
    v2: Float64,
) -> Float64:
    var m = _m_inf(v, v1, v2)
    var i_ca = g_ca * m * (v - e_ca)
    var i_k = g_k * w * (v - e_k)
    var i_l = g_l * (v - e_l)
    return (-i_ca - i_k - i_l + current) / c_m


fn _rhs_w(
    v: Float64,
    w: Float64,
    v3: Float64,
    v4: Float64,
    phi: Float64,
) -> Float64:
    return _lambda(v, v3, v4, phi) * (_w_inf(v, v3, v4) - w)


fn morris_lecar_next_v(
    v: Float64,
    w: Float64,
    current: Float64,
    c_m: Float64,
    g_ca: Float64,
    g_k: Float64,
    g_l: Float64,
    e_ca: Float64,
    e_k: Float64,
    e_l: Float64,
    v1: Float64,
    v2: Float64,
    v3: Float64,
    v4: Float64,
    phi: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Float64:
    if not _finite(current) or not morris_lecar_valid(
        v, w, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2, v3, v4, phi, dt, v_threshold
    ):
        return 0.0 / 0.0
    var k1_v = _rhs_v(v, w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k1_w = _rhs_w(v, w, v3, v4, phi)
    var k2_v = _rhs_v(v + 0.5 * dt * k1_v, w + 0.5 * dt * k1_w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k2_w = _rhs_w(v + 0.5 * dt * k1_v, w + 0.5 * dt * k1_w, v3, v4, phi)
    var k3_v = _rhs_v(v + 0.5 * dt * k2_v, w + 0.5 * dt * k2_w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k3_w = _rhs_w(v + 0.5 * dt * k2_v, w + 0.5 * dt * k2_w, v3, v4, phi)
    var k4_v = _rhs_v(v + dt * k3_v, w + dt * k3_w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var next_v = v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    if not _finite(next_v):
        return 0.0 / 0.0
    return next_v


fn morris_lecar_next_w(
    v: Float64,
    w: Float64,
    current: Float64,
    c_m: Float64,
    g_ca: Float64,
    g_k: Float64,
    g_l: Float64,
    e_ca: Float64,
    e_k: Float64,
    e_l: Float64,
    v1: Float64,
    v2: Float64,
    v3: Float64,
    v4: Float64,
    phi: Float64,
    dt: Float64,
    v_threshold: Float64,
) -> Float64:
    if not _finite(current) or not morris_lecar_valid(
        v, w, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2, v3, v4, phi, dt, v_threshold
    ):
        return 0.0 / 0.0
    var k1_v = _rhs_v(v, w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k1_w = _rhs_w(v, w, v3, v4, phi)
    var k2_v = _rhs_v(v + 0.5 * dt * k1_v, w + 0.5 * dt * k1_w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k2_w = _rhs_w(v + 0.5 * dt * k1_v, w + 0.5 * dt * k1_w, v3, v4, phi)
    var k3_v = _rhs_v(v + 0.5 * dt * k2_v, w + 0.5 * dt * k2_w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2)
    var k3_w = _rhs_w(v + 0.5 * dt * k2_v, w + 0.5 * dt * k2_w, v3, v4, phi)
    var k4_w = _rhs_w(v + dt * k3_v, w + dt * k3_w, v3, v4, phi)
    var next_w = w + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0
    if not _finite(next_w) or next_w < 0.0 or next_w > 1.0:
        return 0.0 / 0.0
    return next_w


fn morris_lecar_step_spike(
    v: Float64,
    next_v: Float64,
    v_threshold: Float64,
) -> Int:
    if not _finite(v) or not _finite(next_v) or not _finite(v_threshold):
        return -1
    if next_v >= v_threshold and v < v_threshold:
        return 1
    return 0


# Run a default neuron (parameters mirroring the Python golden MorrisLecarNeuron) for n_steps
# RK4 steps at constant current and return the rising-edge spike count. This composes the
# per-step primitives above into the simulate() recurrence so the kernel carries a runnable
# parity check.
fn simulate(n_steps: Int, current: Float64) -> Int:
    var v = -60.0
    var w = 0.0
    var c_m = 20.0
    var g_ca = 4.0
    var g_k = 8.0
    var g_l = 2.0
    var e_ca = 120.0
    var e_k = -84.0
    var e_l = -60.0
    var v1 = -1.2
    var v2 = 18.0
    var v3 = 12.0
    var v4 = 17.4
    var phi = 1.0 / 15.0
    var dt = 0.1
    var v_threshold = 0.0
    var spikes: Int = 0
    for _ in range(n_steps):
        var next_v = morris_lecar_next_v(
            v, w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2, v3, v4, phi, dt, v_threshold
        )
        var next_w = morris_lecar_next_w(
            v, w, current, c_m, g_ca, g_k, g_l, e_ca, e_k, e_l, v1, v2, v3, v4, phi, dt, v_threshold
        )
        if morris_lecar_step_spike(v, next_v, v_threshold) > 0:
            spikes += 1
        v = next_v
        w = next_w
    return spikes


def main():
    # Parity contract against the Python golden over 2000 steps: 0 spikes at I=0, 3 at I=50,
    # 5 at I=100 — the same counts the Python, Rust, Go and Julia kernels reproduce. Morris-Lecar
    # gating is tanh/cosh, so the trace is not bit-exact across libms, but the spike count is the
    # stable observable. Run: `mojo run morris_lecar.mojo`.
    var silent = simulate(2000, 0.0)
    print("I=0, 2000 steps -> spikes =", silent, "(expect 0)")
    var three = simulate(2000, 50.0)
    print("I=50, 2000 steps -> spikes =", three, "(expect 3)")
    var five = simulate(2000, 100.0)
    print("I=100, 2000 steps -> spikes =", five, "(expect 5)")
    if silent == 0 and three == 3 and five == 5:
        print("PARITY OK (matches the Python golden across all three regimes)")
    else:
        print("PARITY FAIL: expected 0 / 3 / 5 to match the Python golden")
