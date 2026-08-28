# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Hindmarsh-Rose RK4 simulator (parity with hindmarsh_rose.py)
#
# Build:
#   mojo build --emit shared-lib -o libhr.so hindmarsh_rose.mojo
#
# Parity contract: `hindmarsh_rose_simulate_c` reproduces
# `sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron.simulate`.
# The RK4 right-hand side is exact arithmetic (square `x*x`, cube `(x*x)*x`, no
# transcendental functions). Rust/Julia/Go reproduce the trace bit-for-bit;
# Mojo's release build fuses some of the RK4 multiply-adds into FMAs (one
# rounding instead of two). Unlike the 2-D FitzHugh-Nagumo flow, Hindmarsh-Rose
# is a 3-D chaotic burster, so that per-step ULP is amplified into a divergent
# whole trace and a slightly different spike count — by design, the model's
# sensitive dependence on initial conditions. Mojo is therefore validated on the
# per-step ULP bound and structural invariants, not on the whole trace, matching
# the documented Mojo FMA-parity precedent for wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+3 Float64 slots:
# [0, n) receive the x trace, index n the final x, n+1 the final y, n+2 the final z.
#
# Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87-102.

from std.math import isfinite
from std.memory import UnsafePointer


@always_inline
def _dx(x: Float64, y: Float64, z: Float64, b: Float64, cur: Float64) -> Float64:
    var x2 = x * x
    var x3 = x2 * x
    return y - x3 + b * x2 - z + cur


@always_inline
def _dy(x: Float64, y: Float64) -> Float64:
    var x2 = x * x
    return 1.0 - 5.0 * x2 - y


@always_inline
def _dz(x: Float64, z: Float64, r: Float64, s: Float64, x_rest: Float64) -> Float64:
    return r * (s * (x - x_rest) - z)


@export
def hindmarsh_rose_simulate_c(
    x0: Float64,
    y0: Float64,
    z0: Float64,
    b: Float64,
    r: Float64,
    s: Float64,
    x_rest: Float64,
    dt: Float64,
    x_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or not isfinite(x0)
        or not isfinite(y0)
        or not isfinite(z0)
        or not isfinite(b)
        or not isfinite(r)
        or not isfinite(s)
        or not isfinite(x_rest)
        or not isfinite(dt)
        or not isfinite(x_threshold)
        or not isfinite(current)
        or r <= 0.0
        or s <= 0.0
        or dt <= 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var z = z0
    var dt6 = dt / 6.0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var x_prev = x
        var k1x = _dx(x, y, z, b, current)
        var k1y = _dy(x, y)
        var k1z = _dz(x, z, r, s, x_rest)
        var a2x = x + 0.5 * dt * k1x
        var a2y = y + 0.5 * dt * k1y
        var a2z = z + 0.5 * dt * k1z
        var k2x = _dx(a2x, a2y, a2z, b, current)
        var k2y = _dy(a2x, a2y)
        var k2z = _dz(a2x, a2z, r, s, x_rest)
        var a3x = x + 0.5 * dt * k2x
        var a3y = y + 0.5 * dt * k2y
        var a3z = z + 0.5 * dt * k2z
        var k3x = _dx(a3x, a3y, a3z, b, current)
        var k3y = _dy(a3x, a3y)
        var k3z = _dz(a3x, a3z, r, s, x_rest)
        var a4x = x + dt * k3x
        var a4y = y + dt * k3y
        var a4z = z + dt * k3z
        var k4x = _dx(a4x, a4y, a4z, b, current)
        var k4y = _dy(a4x, a4y)
        var k4z = _dz(a4x, a4z, r, s, x_rest)
        if (
            not isfinite(k1x)
            or not isfinite(k1y)
            or not isfinite(k1z)
            or not isfinite(k2x)
            or not isfinite(k2y)
            or not isfinite(k2z)
            or not isfinite(k3x)
            or not isfinite(k3y)
            or not isfinite(k3z)
            or not isfinite(k4x)
            or not isfinite(k4y)
            or not isfinite(k4z)
        ):
            return -1
        var next_x = x + dt6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        var next_y = y + dt6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        var next_z = z + dt6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        if not isfinite(next_x) or not isfinite(next_y) or not isfinite(next_z):
            return -1
        x = next_x
        y = next_y
        z = next_z
        trace[t] = x
        if x >= x_threshold and x_prev < x_threshold:
            spikes += 1
    if n_steps > 0:
        trace[n_steps] = x
        trace[n_steps + 1] = y
        trace[n_steps + 2] = z
    return spikes
