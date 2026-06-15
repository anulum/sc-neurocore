# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo FitzHugh-Nagumo RK4 simulator (parity with fitzhugh_nagumo.py)
#
# Build:
#   mojo build --emit shared-lib -o libfhn.so fitzhugh_nagumo.mojo
#
# Parity contract: `fitzhugh_nagumo_simulate_c` reproduces
# `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.simulate`.
# The RK4 right-hand side is exact arithmetic (the cube is `v*v*v`, no
# transcendental functions). Rust/Julia/Go reproduce the trace bit-for-bit;
# Mojo's release build contracts some of the RK4 multiply-adds into fused
# multiply-adds (one rounding instead of two), so each step agrees to within a
# couple of ULP. FitzHugh-Nagumo is a two-dimensional flow, so by
# Poincaré-Bendixson it cannot be chaotic and that ULP does not amplify: the
# trace stays within a small band and the spike counts match. This matches the
# documented Mojo FMA-parity precedent for wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final w.
#
# Reference: FitzHugh, R. (1961). Biophys. J. 1:445-466.

from std.memory import UnsafePointer


@always_inline
fn _rhs_v(v: Float64, w: Float64, cur: Float64) -> Float64:
    var v3 = v * v * v
    return v - v3 / 3.0 - w + cur


@always_inline
fn _rhs_w(v: Float64, w: Float64, a: Float64, b: Float64, eps: Float64) -> Float64:
    var bw = b * w
    return eps * (v + a - bw)


@export
fn fitzhugh_nagumo_simulate_c(
    v0: Float64,
    w0: Float64,
    a: Float64,
    b: Float64,
    epsilon: Float64,
    dt: Float64,
    v_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var w = w0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var v_prev = v
        var k1v = _rhs_v(v, w, current)
        var k1w = _rhs_w(v, w, a, b, epsilon)
        var k2v = _rhs_v(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current)
        var k2w = _rhs_w(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, a, b, epsilon)
        var k3v = _rhs_v(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current)
        var k3w = _rhs_w(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, a, b, epsilon)
        var k4v = _rhs_v(v + dt * k3v, w + dt * k3w, current)
        var k4w = _rhs_w(v + dt * k3v, w + dt * k3w, a, b, epsilon)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        trace[t] = v
        if v >= v_threshold and v_prev < v_threshold:
            spikes += 1
    if n_steps > 0:
        trace[n_steps] = v
        trace[n_steps + 1] = w
    return spikes
