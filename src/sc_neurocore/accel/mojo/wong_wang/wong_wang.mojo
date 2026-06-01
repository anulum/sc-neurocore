# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo N-step simulator for the Wong-Wang 2006 decision unit

# Parity contract: wong_wang_simulate_c() produces identical per-step
# traces and final state as the Rust, Julia, and Go kernels given the
# same xi noise buffer (2 samples per step, xi1 then xi2).
#
# Mojo 0.26.2 FFI pattern (per feedback_mojo_026_ffi_pattern):
# @export rejects parametric signatures, so every numpy buffer comes
# in as a raw Int address; reconstruct with `UnsafePointer[Float64,
# MutAnyOrigin](unsafe_from_address=addr)` inside the function body.
#
# Reference: Wong & Wang (2006) J. Neurosci. 26:1314–1328.

from std.math import exp
from std.memory import UnsafePointer

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
    xi1: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    sigma: Float64,
) -> Float64:
    var r1 = phi(j_n * s1 - j_cross * s2 + i_0 + stim1 + sigma * xi1)
    return -s1 / tau_s + (1.0 - s1) * gamma_p * r1


@always_inline
fn derivative_s2(
    s1: Float64,
    s2: Float64,
    stim2: Float64,
    xi2: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    sigma: Float64,
) -> Float64:
    var r2 = phi(j_n * s2 - j_cross * s1 + i_0 + stim2 + sigma * xi2)
    return -s2 / tau_s + (1.0 - s2) * gamma_p * r2


@export
fn wong_wang_simulate_c(
    n: Int,
    s1_init: Float64,
    s2_init: Float64,
    tau_s: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    sigma: Float64,
    dt: Float64,
    stim1_addr: Int,
    stim2_addr: Int,
    xi_addr: Int,
    s1_out_addr: Int,
    s2_out_addr: Int,
    r1_out_addr: Int,
    r2_out_addr: Int,
    s1_final_addr: Int,
    s2_final_addr: Int,
) -> Int:
    var stim1 = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=stim1_addr
    )
    var stim2 = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=stim2_addr
    )
    var xi = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=xi_addr)
    var s1_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=s1_out_addr
    )
    var s2_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=s2_out_addr
    )
    var r1_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=r1_out_addr
    )
    var r2_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=r2_out_addr
    )
    var s1_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=s1_final_addr
    )
    var s2_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=s2_final_addr
    )

    var s1 = s1_init
    var s2 = s2_init

    for t in range(n):
        var xi1 = xi[2 * t]
        var xi2 = xi[2 * t + 1]
        var i1 = j_n * s1 - j_cross * s2 + i_0 + stim1[t] + sigma * xi1
        var i2 = j_n * s2 - j_cross * s1 + i_0 + stim2[t] + sigma * xi2
        var r1 = phi(i1)
        var r2 = phi(i2)
        var k1_s1 = -s1 / tau_s + (1.0 - s1) * gamma_p * r1
        var k1_s2 = -s2 / tau_s + (1.0 - s2) * gamma_p * r2
        var k2_s1 = derivative_s1(
            s1 + 0.5 * dt * k1_s1,
            s2 + 0.5 * dt * k1_s2,
            stim1[t],
            xi1,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        var k2_s2 = derivative_s2(
            s1 + 0.5 * dt * k1_s1,
            s2 + 0.5 * dt * k1_s2,
            stim2[t],
            xi2,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        var k3_s1 = derivative_s1(
            s1 + 0.5 * dt * k2_s1,
            s2 + 0.5 * dt * k2_s2,
            stim1[t],
            xi1,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        var k3_s2 = derivative_s2(
            s1 + 0.5 * dt * k2_s1,
            s2 + 0.5 * dt * k2_s2,
            stim2[t],
            xi2,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        var k4_s1 = derivative_s1(
            s1 + dt * k3_s1,
            s2 + dt * k3_s2,
            stim1[t],
            xi1,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        var k4_s2 = derivative_s2(
            s1 + dt * k3_s1,
            s2 + dt * k3_s2,
            stim2[t],
            xi2,
            tau_s,
            gamma_p,
            j_n,
            j_cross,
            i_0,
            sigma,
        )
        s1 += dt * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1) / 6.0
        s2 += dt * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2) / 6.0
        s1 = clamp01(s1)
        s2 = clamp01(s2)
        s1_out[t] = s1
        s2_out[t] = s2
        r1_out[t] = r1
        r2_out[t] = r2

    s1_final[0] = s1
    s2_final[0] = s2
    return 0
