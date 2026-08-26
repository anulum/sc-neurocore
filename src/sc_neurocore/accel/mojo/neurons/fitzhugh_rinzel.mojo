# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo FitzHugh-Rinzel RK4 simulator (parity with fitzhugh_rinzel.py)
#
# Build:
#   mojo build --emit shared-lib -o libfhr.so fitzhugh_rinzel.mojo
#
# Parity contract: `fitzhugh_rinzel_simulate_c` reproduces
# `sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron.simulate`.
# The RK4 right-hand side is exact arithmetic (cube `v*v*v`, no transcendental
# functions). Rust/Julia/Go reproduce the trace bit-for-bit; Mojo's release
# build fuses some of the RK4 multiply-adds into FMAs (one rounding instead of
# two). With the slow `mu = 1e-4` recovery the dynamics are bursting but not
# strongly chaotic, so the per-step ULP stays a small non-amplifying band and
# the spike counts match. Mojo is validated on the per-step bound, matching the
# documented Mojo FMA-parity precedent for wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+3 Float64 slots:
# [0, n) receive the v trace, index n the final v, n+1 the final w, n+2 the final y.
#
# Reference: FitzHugh, R. (1976); Rinzel, J. (1987).

from std.memory import UnsafePointer


@always_inline
def _dv(v: Float64, w: Float64, y: Float64, cur: Float64) -> Float64:
    var v3 = v * v * v
    return v - v3 / 3.0 - w + y + cur


@always_inline
def _dw(v: Float64, w: Float64, a: Float64, b: Float64, delta: Float64) -> Float64:
    return delta * (a + v - b * w)


@always_inline
def _dy(v: Float64, y: Float64, c: Float64, d: Float64, mu: Float64) -> Float64:
    return mu * (c - v - d * y)


@export
def fitzhugh_rinzel_simulate_c(
    v0: Float64,
    w0: Float64,
    y0: Float64,
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    delta: Float64,
    mu: Float64,
    dt: Float64,
    v_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var w = w0
    var y = y0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var v_prev = v
        var k1v = _dv(v, w, y, current)
        var k1w = _dw(v, w, a, b, delta)
        var k1y = _dy(v, y, c, d, mu)
        var b2v = v + 0.5 * dt * k1v
        var b2w = w + 0.5 * dt * k1w
        var b2y = y + 0.5 * dt * k1y
        var k2v = _dv(b2v, b2w, b2y, current)
        var k2w = _dw(b2v, b2w, a, b, delta)
        var k2y = _dy(b2v, b2y, c, d, mu)
        var b3v = v + 0.5 * dt * k2v
        var b3w = w + 0.5 * dt * k2w
        var b3y = y + 0.5 * dt * k2y
        var k3v = _dv(b3v, b3w, b3y, current)
        var k3w = _dw(b3v, b3w, a, b, delta)
        var k3y = _dy(b3v, b3y, c, d, mu)
        var b4v = v + dt * k3v
        var b4w = w + dt * k3w
        var b4y = y + dt * k3y
        var k4v = _dv(b4v, b4w, b4y, current)
        var k4w = _dw(b4v, b4w, a, b, delta)
        var k4y = _dy(b4v, b4y, c, d, mu)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        trace[t] = v
        if v >= v_threshold and v_prev < v_threshold:
            spikes += 1
    if n_steps > 0:
        trace[n_steps] = v
        trace[n_steps + 1] = w
        trace[n_steps + 2] = y
    return spikes
