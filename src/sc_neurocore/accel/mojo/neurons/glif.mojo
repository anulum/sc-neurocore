# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Allen GLIF5 (parity with glif.py)
#
# Build:
#   mojo build --emit shared-lib -o libglif.so glif.mojo
#
# Parity contract: `glif_simulate_c` reproduces
# `sc_neurocore.neurons.models.glif.GLIFNeuron.simulate`. The Allen GLIF5
# right-hand side is purely linear; each product is rounded into its own variable
# before the following add/subtract to limit FMA contraction. The model is
# validated per-step and on spike counts rather than bit-exactly, because Mojo
# fuses multiply-add.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+4 Float64 slots:
# [0, n) receive the v trace, indices n..n+3 the final (v, theta, i_asc1, i_asc2).
#
# Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.

from std.memory import UnsafePointer


@always_inline
def _dv(
    v: Float64,
    a1: Float64,
    a2: Float64,
    current: Float64,
    v_rest: Float64,
    resistance: Float64,
    tau_m: Float64,
) -> Float64:
    var drive = resistance * current
    return (-(v - v_rest) + drive + a1 + a2) / tau_m


@always_inline
def _dtheta(
    v: Float64,
    theta: Float64,
    theta_inf: Float64,
    a_theta: Float64,
    v_rest: Float64,
    tau_theta: Float64,
) -> Float64:
    var coupling = a_theta * (v - v_rest)
    return (theta_inf - theta + coupling) / tau_theta


@export
def glif_simulate_c(
    v0: Float64,
    theta0: Float64,
    theta_inf: Float64,
    i_asc1_0: Float64,
    i_asc2_0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    tau_asc1: Float64,
    tau_asc2: Float64,
    a_theta: Float64,
    delta_theta: Float64,
    r_asc1: Float64,
    r_asc2: Float64,
    resistance: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var theta = theta0
    var a1 = i_asc1_0
    var a2 = i_asc2_0
    var half_dt = 0.5 * dt
    var spikes: Int64 = 0
    for t in range(n_steps):
        var k1v = _dv(v, a1, a2, current, v_rest, resistance, tau_m)
        var k1t = _dtheta(v, theta, theta_inf, a_theta, v_rest, tau_theta)
        var k1a = -a1 / tau_asc1
        var k1b = -a2 / tau_asc2

        var v2 = v + half_dt * k1v
        var th2 = theta + half_dt * k1t
        var a1_2 = a1 + half_dt * k1a
        var a2_2 = a2 + half_dt * k1b
        var k2v = _dv(v2, a1_2, a2_2, current, v_rest, resistance, tau_m)
        var k2t = _dtheta(v2, th2, theta_inf, a_theta, v_rest, tau_theta)
        var k2a = -a1_2 / tau_asc1
        var k2b = -a2_2 / tau_asc2

        var v3 = v + half_dt * k2v
        var th3 = theta + half_dt * k2t
        var a1_3 = a1 + half_dt * k2a
        var a2_3 = a2 + half_dt * k2b
        var k3v = _dv(v3, a1_3, a2_3, current, v_rest, resistance, tau_m)
        var k3t = _dtheta(v3, th3, theta_inf, a_theta, v_rest, tau_theta)
        var k3a = -a1_3 / tau_asc1
        var k3b = -a2_3 / tau_asc2

        var v4 = v + dt * k3v
        var th4 = theta + dt * k3t
        var a1_4 = a1 + dt * k3a
        var a2_4 = a2 + dt * k3b
        var k4v = _dv(v4, a1_4, a2_4, current, v_rest, resistance, tau_m)
        var k4t = _dtheta(v4, th4, theta_inf, a_theta, v_rest, tau_theta)
        var k4a = -a1_4 / tau_asc1
        var k4b = -a2_4 / tau_asc2

        var sv = k1v + 2.0 * k2v + 2.0 * k3v + k4v
        var st = k1t + 2.0 * k2t + 2.0 * k3t + k4t
        var sa = k1a + 2.0 * k2a + 2.0 * k3a + k4a
        var sb = k1b + 2.0 * k2b + 2.0 * k3b + k4b
        v = v + dt * sv / 6.0
        theta = theta + dt * st / 6.0
        a1 = a1 + dt * sa / 6.0
        a2 = a2 + dt * sb / 6.0

        if v >= theta:
            v = v_reset
            theta += delta_theta
            a1 += r_asc1
            a2 += r_asc2
            spikes += 1
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = theta
    trace[n_steps + 2] = a1
    trace[n_steps + 3] = a2
    return spikes
