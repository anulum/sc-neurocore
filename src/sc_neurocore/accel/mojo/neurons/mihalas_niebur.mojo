# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Mihalas-Niebur 2009 generalised IF (parity with mihalas_niebur.py)
#
# Build:
#   mojo build --emit shared-lib -o libmihalasniebur.so mihalas_niebur.mojo
#
# Parity contract: `mihalas_niebur_simulate_c` reproduces
# `sc_neurocore.neurons.models.mihalas_niebur.MihalasNieburNeuron.simulate`. The
# Mihalas-Niebur right-hand side is purely linear; each product is rounded into
# its own variable before the following add/subtract to limit FMA contraction.
# The model is validated per-step and on spike counts rather than bit-exactly,
# because Mojo fuses multiply-add.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+4 Float64 slots:
# [0, n) receive the v trace, indices n..n+3 the final (v, theta, i1, i2).
#
# Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.

from std.memory import UnsafePointer


@always_inline
def _dv(
    v: Float64, i1: Float64, i2: Float64, current: Float64, v_rest: Float64, tau_v: Float64
) -> Float64:
    return (-(v - v_rest) + i1 + i2 + current) / tau_v


@always_inline
def _dtheta(
    v: Float64,
    theta: Float64,
    theta_inf: Float64,
    a: Float64,
    v_rest: Float64,
    tau_theta: Float64,
) -> Float64:
    var coupling = a * (v - v_rest)
    return (theta_inf - theta + coupling) / tau_theta


@export
def mihalas_niebur_simulate_c(
    v0: Float64,
    theta0: Float64,
    i1_0: Float64,
    i2_0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_reset: Float64,
    theta_inf: Float64,
    tau_v: Float64,
    tau_theta: Float64,
    tau_1: Float64,
    tau_2: Float64,
    a: Float64,
    b: Float64,
    r1: Float64,
    r2: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var theta = theta0
    var i1 = i1_0
    var i2 = i2_0
    var half_dt = 0.5 * dt
    var spikes: Int64 = 0
    for t in range(n_steps):
        var k1v = _dv(v, i1, i2, current, v_rest, tau_v)
        var k1t = _dtheta(v, theta, theta_inf, a, v_rest, tau_theta)
        var k1a = -i1 / tau_1
        var k1b = -i2 / tau_2

        var v2 = v + half_dt * k1v
        var th2 = theta + half_dt * k1t
        var i1_2 = i1 + half_dt * k1a
        var i2_2 = i2 + half_dt * k1b
        var k2v = _dv(v2, i1_2, i2_2, current, v_rest, tau_v)
        var k2t = _dtheta(v2, th2, theta_inf, a, v_rest, tau_theta)
        var k2a = -i1_2 / tau_1
        var k2b = -i2_2 / tau_2

        var v3 = v + half_dt * k2v
        var th3 = theta + half_dt * k2t
        var i1_3 = i1 + half_dt * k2a
        var i2_3 = i2 + half_dt * k2b
        var k3v = _dv(v3, i1_3, i2_3, current, v_rest, tau_v)
        var k3t = _dtheta(v3, th3, theta_inf, a, v_rest, tau_theta)
        var k3a = -i1_3 / tau_1
        var k3b = -i2_3 / tau_2

        var v4 = v + dt * k3v
        var th4 = theta + dt * k3t
        var i1_4 = i1 + dt * k3a
        var i2_4 = i2 + dt * k3b
        var k4v = _dv(v4, i1_4, i2_4, current, v_rest, tau_v)
        var k4t = _dtheta(v4, th4, theta_inf, a, v_rest, tau_theta)
        var k4a = -i1_4 / tau_1
        var k4b = -i2_4 / tau_2

        var sv = k1v + 2.0 * k2v + 2.0 * k3v + k4v
        var st = k1t + 2.0 * k2t + 2.0 * k3t + k4t
        var sa = k1a + 2.0 * k2a + 2.0 * k3a + k4a
        var sb = k1b + 2.0 * k2b + 2.0 * k3b + k4b
        v = v + dt * sv / 6.0
        theta = theta + dt * st / 6.0
        i1 = i1 + dt * sa / 6.0
        i2 = i2 + dt * sb / 6.0

        if v >= theta:
            v = v_reset + b * (v - v_rest)
            if theta_reset > theta:
                theta = theta_reset
            i1 += r1
            i2 += r2
            spikes += 1
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = theta
    trace[n_steps + 2] = i1
    trace[n_steps + 3] = i2
    return spikes
