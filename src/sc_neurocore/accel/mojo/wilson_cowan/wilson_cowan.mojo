# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic Mojo Wilson-Cowan RK4 batch and C ABI

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def logistic(z: Float64) -> Float64:
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    var exp_z = exp(z)
    return exp_z / (1.0 + exp_z)


@always_inline
def sigmoid(a: Float64, theta: Float64, x: Float64) -> Float64:
    return logistic(a * (x - theta)) - logistic(-a * theta)


@always_inline
def finite_rate(value: Float64, a: Float64, theta: Float64) -> Bool:
    var baseline = logistic(-a * theta)
    return isfinite(value) and value >= -baseline and value <= 1.0


@always_inline
def valid_configuration(
    e: Float64,
    i: Float64,
    w_ee: Float64,
    w_ei: Float64,
    w_ie: Float64,
    w_ii: Float64,
    tau_e: Float64,
    tau_i: Float64,
    a: Float64,
    theta: Float64,
    dt: Float64,
) -> Bool:
    return (
        finite_rate(e, a, theta)
        and finite_rate(i, a, theta)
        and isfinite(w_ee)
        and w_ee >= 0.0
        and isfinite(w_ei)
        and w_ei >= 0.0
        and isfinite(w_ie)
        and w_ie >= 0.0
        and isfinite(w_ii)
        and w_ii >= 0.0
        and isfinite(tau_e)
        and tau_e > 0.0
        and isfinite(tau_i)
        and tau_i > 0.0
        and isfinite(a)
        and a > 0.0
        and isfinite(theta)
        and isfinite(dt)
        and dt > 0.0
    )


@always_inline
def derivative_e(
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
def derivative_i(
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


def _run_wilson_cowan(
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
    write_output: Bool,
) -> Int:
    if n < 0 or e_final_addr == 0 or i_final_addr == 0:
        return -1
    if n > 0 and (ext_addr == 0 or e_out_addr == 0 or i_out_addr == 0):
        return -1
    if not valid_configuration(
        e_init,
        i_init,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
    ):
        return -1

    if n == 0:
        if write_output:
            var empty_ef = UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=e_final_addr
            )
            var empty_if = UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=i_final_addr
            )
            empty_ef[0] = e_init
            empty_if[0] = i_init
        return 0

    var ext = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=ext_addr)
    var eo = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=e_out_addr)
    var io = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=i_out_addr)
    var ef = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=e_final_addr)
    var iff = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=i_final_addr)

    var e = e_init
    var i = i_init
    for t in range(n):
        var drive = ext[t]
        if not isfinite(drive):
            return -1
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
        if not (
            isfinite(k1_e)
            and isfinite(k1_i)
            and isfinite(k2_e)
            and isfinite(k2_i)
            and isfinite(k3_e)
            and isfinite(k3_i)
            and isfinite(k4_e)
            and isfinite(k4_i)
        ):
            return -1
        var next_e = e + dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0
        var next_i = i + dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0
        if not finite_rate(next_e, a, theta) or not finite_rate(next_i, a, theta):
            return -1
        e = next_e
        i = next_i
        if write_output:
            eo[t] = e
            io[t] = i
    if write_output:
        ef[0] = e
        iff[0] = i
    return 0


@export
def wilson_cowan_simulate_c(
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
    var validated = _run_wilson_cowan(
        n,
        e_init,
        i_init,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        ext_addr,
        e_out_addr,
        i_out_addr,
        e_final_addr,
        i_final_addr,
        False,
    )
    if validated != 0:
        return -1
    return _run_wilson_cowan(
        n,
        e_init,
        i_init,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        ext_addr,
        e_out_addr,
        i_out_addr,
        e_final_addr,
        i_final_addr,
        True,
    )
