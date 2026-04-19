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
    var eo = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=e_out_addr)
    var io = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=i_out_addr)
    var ef = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=e_final_addr
    )
    var iff = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_final_addr
    )

    var e = e_init
    var i = i_init
    for t in range(n):
        var s_e = sigmoid(a, theta, w_ee * e - w_ei * i + ext[t])
        var s_i = sigmoid(a, theta, w_ie * e - w_ii * i)
        e += (-e + s_e) / tau_e * dt
        i += (-i + s_i) / tau_i * dt
        eo[t] = e
        io[t] = i
    ef[0] = e
    iff[0] = i
    return 0
