# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo N-step simulator for the Wilson-Cowan 1972 E/I model

# Parity contract: wilson_cowan_simulate_c() produces identical per-step
# traces as the Rust, Julia, and Go kernels under f64 arithmetic
# (Wilson-Cowan is deterministic — no noise).
#
# Mojo 0.26.2 FFI pattern (per feedback_mojo_026_ffi_pattern):
# @export rejects parametric signatures, so every numpy buffer comes
# in as a raw `Int` address; reconstruct with
# `UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)`
# inside the function body.
#
# Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.

from std.math import exp
from std.memory import UnsafePointer


@always_inline
fn sigmoid(a: Float64, theta: Float64, x: Float64) -> Float64:
    # Published Wilson-Cowan 1972 two-term sigmoid:
    #   S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    var baseline = 1.0 / (1.0 + exp(a * theta))
    return 1.0 / (1.0 + exp(-a * (x - theta))) - baseline


@always_inline
fn derivative_e(
    e: Float64,
    i: Float64,
    ext: Float64,
    w_ee: Float64,
    w_ei: Float64,
    tau_e: Float64,
    a: Float64,
    theta: Float64,
) -> Float64:
    var s_e = sigmoid(a, theta, w_ee * e - w_ei * i + ext)
    return (-e + s_e) / tau_e


@always_inline
fn derivative_i(
    e: Float64,
    i: Float64,
    w_ie: Float64,
    w_ii: Float64,
    tau_i: Float64,
    a: Float64,
    theta: Float64,
) -> Float64:
    var s_i = sigmoid(a, theta, w_ie * e - w_ii * i)
    return (-i + s_i) / tau_i


@export
fn wilson_cowan_simulate_c(
    n: Int,
    e_init: Float64,
    i_init: Float64,
    w_ee: Float64,
    w_ei: Float64,
    w_ie: Float64,
    w_ii: Float64,
    tau_e: Float64,
    tau_i: Float64,
    a: Float64,
    theta: Float64,
    dt: Float64,
    ext_addr: Int,
    e_out_addr: Int,
    i_out_addr: Int,
    e_final_addr: Int,
    i_final_addr: Int,
) -> Int:
    var ext = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=ext_addr)
    var eo = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=e_out_addr
    )
    var io = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_out_addr
    )
    var ef = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=e_final_addr
    )
    var iff = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_final_addr
    )

    var e = e_init
    var i = i_init
    for t in range(n):
        var drive = ext[t]
        var k1_e = derivative_e(e, i, drive, w_ee, w_ei, tau_e, a, theta)
        var k1_i = derivative_i(e, i, w_ie, w_ii, tau_i, a, theta)
        var k2_e = derivative_e(
            e + 0.5 * dt * k1_e,
            i + 0.5 * dt * k1_i,
            drive,
            w_ee,
            w_ei,
            tau_e,
            a,
            theta,
        )
        var k2_i = derivative_i(
            e + 0.5 * dt * k1_e,
            i + 0.5 * dt * k1_i,
            w_ie,
            w_ii,
            tau_i,
            a,
            theta,
        )
        var k3_e = derivative_e(
            e + 0.5 * dt * k2_e,
            i + 0.5 * dt * k2_i,
            drive,
            w_ee,
            w_ei,
            tau_e,
            a,
            theta,
        )
        var k3_i = derivative_i(
            e + 0.5 * dt * k2_e,
            i + 0.5 * dt * k2_i,
            w_ie,
            w_ii,
            tau_i,
            a,
            theta,
        )
        var k4_e = derivative_e(
            e + dt * k3_e,
            i + dt * k3_i,
            drive,
            w_ee,
            w_ei,
            tau_e,
            a,
            theta,
        )
        var k4_i = derivative_i(
            e + dt * k3_e,
            i + dt * k3_i,
            w_ie,
            w_ii,
            tau_i,
            a,
            theta,
        )
        e += dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0
        i += dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0
        eo[t] = e
        io[t] = i
    ef[0] = e
    iff[0] = i
    return 0
