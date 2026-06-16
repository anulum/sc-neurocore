# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Terman-Wang 1995 LEGION relaxation oscillator (parity with terman_wang.py)
#
# Build:
#   mojo build --emit shared-lib -o libtermanwang.so terman_wang.mojo
#
# Parity contract: `terman_wang_simulate_c` reproduces
# `sc_neurocore.neurons.models.terman_wang.TermanWangOscillator.simulate`. The
# cubic is exact (v*v*v); each product is rounded into its own variable before the
# following add/subtract to avoid FMA contraction, and the `tanh` gating uses
# Mojo's libm. The two-dimensional relaxation oscillator is non-chaotic, so the
# residual transcendental/FMA ULP stays bounded; the backend is validated
# per-step and on spike counts.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final w.
#
# Reference: Terman, D. & Wang, D.L. (1995). Physica D 81:148-176.

from math import tanh
from std.memory import UnsafePointer


@always_inline
fn _dv(v: Float64, w: Float64, current: Float64, rho: Float64) -> Float64:
    var cube = v * v * v
    var f = 3.0 * v - cube + 2.0
    return f - w + current + rho


@always_inline
fn _dw(v: Float64, w: Float64, alpha: Float64, beta: Float64, eps: Float64) -> Float64:
    var arg = v / beta
    var g = alpha * (1.0 + tanh(arg))
    return eps * (g - w)


@export
fn terman_wang_simulate_c(
    v0: Float64,
    w0: Float64,
    alpha: Float64,
    beta: Float64,
    eps: Float64,
    rho: Float64,
    dt: Float64,
    v_peak: Float64,
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
        var hd = 0.5 * dt
        var dv1 = _dv(v, w, current, rho)
        var dw1 = _dw(v, w, alpha, beta, eps)
        var v2 = v + hd * dv1
        var w2 = w + hd * dw1
        var dv2 = _dv(v2, w2, current, rho)
        var dw2 = _dw(v2, w2, alpha, beta, eps)
        var v3 = v + hd * dv2
        var w3 = w + hd * dw2
        var dv3 = _dv(v3, w3, current, rho)
        var dw3 = _dw(v3, w3, alpha, beta, eps)
        var v4 = v + dt * dv3
        var w4 = w + dt * dw3
        var dv4 = _dv(v4, w4, current, rho)
        var dw4 = _dw(v4, w4, alpha, beta, eps)
        var sv = dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4
        var sw = dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4
        v = v + dt * sv / 6.0
        w = w + dt * sw / 6.0
        trace[t] = v
        if v >= v_peak and v_prev < v_peak:
            spikes += 1
    trace[n_steps] = v
    trace[n_steps + 1] = w
    return spikes
