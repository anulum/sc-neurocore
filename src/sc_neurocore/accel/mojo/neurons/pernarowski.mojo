# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Pernarowski 1994 beta-cell burster (parity with pernarowski.py)
#
# Build:
#   mojo build --emit shared-lib -o libpernarowski.so pernarowski.mojo
#
# Parity contract: `pernarowski_simulate_c` reproduces
# `sc_neurocore.neurons.models.pernarowski.PernarowskiNeuron.simulate`. The
# polynomial RHS is exact arithmetic (cubic v*v*v); each product is rounded into
# its own variable before the following add/subtract so the compiler cannot
# contract a multiply-add into a single-rounding FMA — the one operation that
# diverges from the IEEE-754 two-rounding path used by Python/Rust/Go/Julia. The
# model is a slow-fast burster (not chaotic), so any residual single-ULP
# difference stays bounded; the backend is validated per-step and on spike counts.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+3 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 w, index n+2 z.
#
# Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814-832.

from std.memory import UnsafePointer


@always_inline
def _dv(v: Float64, w: Float64, z: Float64, current: Float64) -> Float64:
    var cube = v * v * v
    var third = cube / 3.0
    return v - third - w - z + current


@always_inline
def _dw(v: Float64, w: Float64, eps1: Float64, gamma: Float64, alpha: Float64) -> Float64:
    var gw = gamma * w
    return eps1 * (v - gw + alpha)


@always_inline
def _dz(v: Float64, z: Float64, eps2: Float64, beta: Float64) -> Float64:
    var shifted = v + 0.7
    var bv = beta * shifted
    return eps2 * (bv - z)


@export
def pernarowski_simulate_c(
    v0: Float64,
    w0: Float64,
    z0: Float64,
    alpha: Float64,
    beta: Float64,
    eps1: Float64,
    eps2: Float64,
    gamma: Float64,
    dt: Float64,
    v_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var w = w0
    var z = z0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var v_prev = v
        var hd = 0.5 * dt
        var dv1 = _dv(v, w, z, current)
        var dw1 = _dw(v, w, eps1, gamma, alpha)
        var dz1 = _dz(v, z, eps2, beta)
        var v2 = v + hd * dv1
        var w2 = w + hd * dw1
        var z2 = z + hd * dz1
        var dv2 = _dv(v2, w2, z2, current)
        var dw2 = _dw(v2, w2, eps1, gamma, alpha)
        var dz2 = _dz(v2, z2, eps2, beta)
        var v3 = v + hd * dv2
        var w3 = w + hd * dw2
        var z3 = z + hd * dz2
        var dv3 = _dv(v3, w3, z3, current)
        var dw3 = _dw(v3, w3, eps1, gamma, alpha)
        var dz3 = _dz(v3, z3, eps2, beta)
        var v4 = v + dt * dv3
        var w4 = w + dt * dw3
        var z4 = z + dt * dz3
        var dv4 = _dv(v4, w4, z4, current)
        var dw4 = _dw(v4, w4, eps1, gamma, alpha)
        var dz4 = _dz(v4, z4, eps2, beta)
        var sv = dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4
        var sw = dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4
        var sz = dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4
        v = v + dt * sv / 6.0
        w = w + dt * sw / 6.0
        z = z + dt * sz / 6.0
        if v >= v_threshold and v_prev < v_threshold:
            spikes += 1
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = w
    trace[n_steps + 2] = z
    return spikes
