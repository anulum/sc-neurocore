# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Izhikevich 2007 RK4 simulator (parity with izhikevich2007.py)
#
# Build:
#   mojo build --emit shared-lib -o libizh2007.so izhikevich2007.mojo
#
# Parity contract: `izhikevich2007_simulate_c` reproduces
# `sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron.simulate`.
# The NeuroML right-hand side `k (v-vr)(v-vt)/C` is exact arithmetic (products, a
# sum, a division — no transcendental functions). Rust/Julia/Go reproduce the
# trace bit-for-bit; Mojo's release build fuses some of the RK4 multiply-adds
# into FMAs (one rounding instead of two). The hard `v >= vpeak -> v = c` reset
# re-anchors the trajectory every spike, so the per-step ULP stays a small
# non-amplifying band and the spike counts match. Mojo is validated on that
# bound, matching the documented Mojo FMA-parity precedent for wong_wang.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final u.
#
# Reference: Izhikevich, E.M. (2007), Dynamical Systems in Neuroscience.

from std.math import isfinite
from std.memory import UnsafePointer


@always_inline
def _dv(
    v: Float64,
    u: Float64,
    cap: Float64,
    k: Float64,
    vr: Float64,
    vt: Float64,
    cur: Float64,
) -> Float64:
    var dvr = v - vr
    var dvt = v - vt
    var quad = k * dvr * dvt
    return (quad - u + cur) / cap


@always_inline
def _du(v: Float64, u: Float64, vr: Float64, a: Float64, b: Float64) -> Float64:
    var bv = b * (v - vr)
    return a * (bv - u)


@export
def izhikevich2007_simulate_c(
    v0: Float64,
    u0: Float64,
    cap: Float64,
    k: Float64,
    vr: Float64,
    vt: Float64,
    vpeak: Float64,
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(v0)
        or not isfinite(u0)
        or not isfinite(cap)
        or cap <= 0.0
        or not isfinite(k)
        or not isfinite(vr)
        or not isfinite(vt)
        or not isfinite(vpeak)
        or not isfinite(a)
        or not isfinite(b)
        or not isfinite(c)
        or not isfinite(d)
        or not isfinite(dt)
        or dt <= 0.0
        or not isfinite(current)
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=trace_addr
    )
    var v = v0
    var u = u0
    var dt6 = dt / 6.0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var k1v = _dv(v, u, cap, k, vr, vt, current)
        var k1u = _du(v, u, vr, a, b)
        var a2v = v + 0.5 * dt * k1v
        var a2u = u + 0.5 * dt * k1u
        var k2v = _dv(a2v, a2u, cap, k, vr, vt, current)
        var k2u = _du(a2v, a2u, vr, a, b)
        var a3v = v + 0.5 * dt * k2v
        var a3u = u + 0.5 * dt * k2u
        var k3v = _dv(a3v, a3u, cap, k, vr, vt, current)
        var k3u = _du(a3v, a3u, vr, a, b)
        var a4v = v + dt * k3v
        var a4u = u + dt * k3u
        var k4v = _dv(a4v, a4u, cap, k, vr, vt, current)
        var k4u = _du(a4v, a4u, vr, a, b)
        v = v + dt6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + dt6 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        if v >= vpeak:
            v = c
            u = u + d
            spikes += 1
        if not isfinite(v) or not isfinite(u):
            return -2
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = u
    return spikes
