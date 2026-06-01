# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar helpers for wong_wang

from std.math import exp

comptime A = 270.0
comptime B = 108.0
comptime D = 0.154


@always_inline
fn phi(i_syn: Float64) -> Float64:
    var x = A * i_syn - B
    if abs(x) < 1e-6:
        return 1.0 / D
    return x / (1.0 - exp(-D * x))


@always_inline
fn clamp01(x: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@always_inline
fn derivative_s1(
    s1: Float64,
    s2: Float64,
    stim1: Float64,
    noise1: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
) -> Float64:
    var r1 = phi(j_n * s1 - j_cross * s2 + i_0 + stim1 + noise1)
    return -s1 / tau_s + (1.0 - s1) * gamma_p * r1


@always_inline
fn derivative_s2(
    s1: Float64,
    s2: Float64,
    stim2: Float64,
    noise2: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
) -> Float64:
    var r2 = phi(j_n * s2 - j_cross * s1 + i_0 + stim2 + noise2)
    return -s2 / tau_s + (1.0 - s2) * gamma_p * r2


fn step_rk4_s1(
    s1: Float64,
    s2: Float64,
    stim1: Float64,
    stim2: Float64,
    noise1: Float64,
    noise2: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    dt: Float64,
) -> Float64:
    var k1_s1 = derivative_s1(
        s1, s2, stim1, noise1, tau_s, gamma_p, j_n, j_cross, i_0
    )
    var k1_s2 = derivative_s2(
        s1, s2, stim2, noise2, tau_s, gamma_p, j_n, j_cross, i_0
    )
    var k2_s1 = derivative_s1(
        s1 + 0.5 * dt * k1_s1,
        s2 + 0.5 * dt * k1_s2,
        stim1,
        noise1,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k2_s2 = derivative_s2(
        s1 + 0.5 * dt * k1_s1,
        s2 + 0.5 * dt * k1_s2,
        stim2,
        noise2,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k3_s1 = derivative_s1(
        s1 + 0.5 * dt * k2_s1,
        s2 + 0.5 * dt * k2_s2,
        stim1,
        noise1,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k3_s2 = derivative_s2(
        s1 + 0.5 * dt * k2_s1,
        s2 + 0.5 * dt * k2_s2,
        stim2,
        noise2,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k4_s1 = derivative_s1(
        s1 + dt * k3_s1,
        s2 + dt * k3_s2,
        stim1,
        noise1,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    return clamp01(s1 + dt * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1) / 6.0)


fn step_rk4_s2(
    s1: Float64,
    s2: Float64,
    stim1: Float64,
    stim2: Float64,
    noise1: Float64,
    noise2: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    dt: Float64,
) -> Float64:
    var k1_s1 = derivative_s1(
        s1, s2, stim1, noise1, tau_s, gamma_p, j_n, j_cross, i_0
    )
    var k1_s2 = derivative_s2(
        s1, s2, stim2, noise2, tau_s, gamma_p, j_n, j_cross, i_0
    )
    var k2_s1 = derivative_s1(
        s1 + 0.5 * dt * k1_s1,
        s2 + 0.5 * dt * k1_s2,
        stim1,
        noise1,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k2_s2 = derivative_s2(
        s1 + 0.5 * dt * k1_s1,
        s2 + 0.5 * dt * k1_s2,
        stim2,
        noise2,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k3_s1 = derivative_s1(
        s1 + 0.5 * dt * k2_s1,
        s2 + 0.5 * dt * k2_s2,
        stim1,
        noise1,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k3_s2 = derivative_s2(
        s1 + 0.5 * dt * k2_s1,
        s2 + 0.5 * dt * k2_s2,
        stim2,
        noise2,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    var k4_s2 = derivative_s2(
        s1 + dt * k3_s1,
        s2 + dt * k3_s2,
        stim2,
        noise2,
        tau_s,
        gamma_p,
        j_n,
        j_cross,
        i_0,
    )
    return clamp01(s2 + dt * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2) / 6.0)


fn reset_s1() -> Float64:
    return 0.1


fn reset_s2() -> Float64:
    return 0.1
