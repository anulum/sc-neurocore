# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo batch mirror for Wong-Wang 2006

from std.math import exp, isfinite, sqrt
from std.memory import UnsafePointer

comptime A = 270.0
comptime B = 108.0
comptime D = 0.154


@always_inline
def phi(i_syn: Float64) -> Float64:
    var x = A * i_syn - B
    var scaled = -D * x
    if scaled > 700.0:
        return 0.0
    if abs(x) < 1.0e-7:
        return 1.0 / D
    var response = x / (1.0 - exp(scaled))
    if response < 0.0:
        return 0.0
    return response


@always_inline
def finite_gate(value: Float64) -> Bool:
    return isfinite(value) and value >= 0.0 and value <= 1.0


@always_inline
def valid_configuration(
    s1: Float64,
    s2: Float64,
    noise1: Float64,
    noise2: Float64,
    tau_s: Float64,
    tau_ampa: Float64,
    gamma_p: Float64,
    j_n: Float64,
    j_cross: Float64,
    i_0: Float64,
    sigma: Float64,
    dt: Float64,
) -> Bool:
    return (
        finite_gate(s1)
        and finite_gate(s2)
        and isfinite(noise1)
        and isfinite(noise2)
        and isfinite(tau_s)
        and tau_s > 0.0
        and isfinite(tau_ampa)
        and tau_ampa > 0.0
        and isfinite(gamma_p)
        and gamma_p > 0.0
        and isfinite(j_n)
        and j_n >= 0.0
        and isfinite(j_cross)
        and j_cross >= 0.0
        and isfinite(i_0)
        and isfinite(sigma)
        and sigma >= 0.0
        and isfinite(dt)
        and dt > 0.0
    )


@export
def wong_wang_simulate_c(
    n: Int,
    s1_init: Float64,
    s2_init: Float64,
    noise1_init: Float64,
    noise2_init: Float64,
    tau_s: Float64,
    tau_ampa: Float64,
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
    noise1_out_addr: Int,
    noise2_out_addr: Int,
    r1_out_addr: Int,
    r2_out_addr: Int,
    s1_final_addr: Int,
    s2_final_addr: Int,
    noise1_final_addr: Int,
    noise2_final_addr: Int,
) -> Int:
    if n < 0:
        return 1
    if not valid_configuration(
        s1_init,
        s2_init,
        noise1_init,
        noise2_init,
        tau_s,
        tau_ampa,
        gamma_p,
        j_n,
        j_cross,
        i_0,
        sigma,
        dt,
    ):
        return 2

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
    var noise1_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=noise1_out_addr
    )
    var noise2_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=noise2_out_addr
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
    var noise1_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=noise1_final_addr
    )
    var noise2_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=noise2_final_addr
    )

    var s1 = s1_init
    var s2 = s2_init
    var noise1 = noise1_init
    var noise2 = noise2_init
    var noise_scale = sqrt(dt / tau_ampa) * sigma
    var noise_decay = dt / tau_ampa
    for step in range(n):
        var drive1 = stim1[step]
        var drive2 = stim2[step]
        var xi1 = xi[2 * step]
        var xi2 = xi[2 * step + 1]
        if not (
            isfinite(drive1)
            and isfinite(drive2)
            and isfinite(xi1)
            and isfinite(xi2)
        ):
            return 3
        var rate1 = phi(j_n * s1 - j_cross * s2 + i_0 + drive1 + noise1)
        var rate2 = phi(j_n * s2 - j_cross * s1 + i_0 + drive2 + noise2)
        if not (isfinite(rate1) and isfinite(rate2)):
            return 4
        var next_s1 = s1 + dt * (-s1 / tau_s + (1.0 - s1) * gamma_p * rate1)
        var next_s2 = s2 + dt * (-s2 / tau_s + (1.0 - s2) * gamma_p * rate2)
        var next_noise1 = noise1 - noise_decay * noise1 + noise_scale * xi1
        var next_noise2 = noise2 - noise_decay * noise2 + noise_scale * xi2
        if not (
            finite_gate(next_s1)
            and finite_gate(next_s2)
            and isfinite(next_noise1)
            and isfinite(next_noise2)
        ):
            return 5
        s1 = next_s1
        s2 = next_s2
        noise1 = next_noise1
        noise2 = next_noise2
        s1_out[step] = s1
        s2_out[step] = s2
        noise1_out[step] = noise1
        noise2_out[step] = noise2
        r1_out[step] = rate1
        r2_out[step] = rate2

    s1_final[0] = s1
    s2_final[0] = s2
    noise1_final[0] = noise1
    noise2_final[0] = noise2
    return 0
