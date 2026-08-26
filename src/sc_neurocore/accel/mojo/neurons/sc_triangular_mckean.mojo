# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained triangular McKean-like RK4 shared library
#
# Build:
#   mojo build --emit shared-lib -o libsc_triangular_mckean.so sc_triangular_mckean.mojo
#
# Parity contract: `sc_triangular_mckean_simulate_c` reproduces
# `sc_neurocore.neurons.models.sc_triangular_mckean.SCTriangularMcKeanNeuron.simulate`. The piecewise-linear
# RHS is exact arithmetic; each product is rounded into its own variable before
# the following add/subtract so the compiler cannot contract a multiply-add into a
# single-rounding FMA — that fusion is the one operation that diverges from the
# IEEE-754 two-rounding path used by Python/Rust/Go/Julia. The two-dimensional
# autonomous flow cannot be chaotic, so any residual single-ULP difference does
# not amplify; the backend is validated per-step and on spike counts.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final w.
#
# SC project recurrence; no external paper attribution.

from std.memory import UnsafePointer


@always_inline
def _fv(x: Float64, half_a: Float64, mid: Float64, a: Float64) -> Float64:
    if x < half_a:
        return -x
    if x < mid:
        return x - a
    return 1.0 - x


@export
def sc_triangular_mckean_simulate_c(
    v0: Float64,
    w0: Float64,
    a: Float64,
    eps: Float64,
    gamma: Float64,
    dt: Float64,
    v_peak: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var w = w0
    var half_a = a / 2.0
    var mid = (1.0 + a) / 2.0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var v_prev = v
        # k1
        var dv1 = _fv(v, half_a, mid, a) - w + current
        var gw1 = gamma * w
        var dw1 = eps * (v - gw1)
        # k2
        var hd1 = 0.5 * dt
        var v2 = v + hd1 * dv1
        var w2 = w + hd1 * dw1
        var dv2 = _fv(v2, half_a, mid, a) - w2 + current
        var gw2 = gamma * w2
        var dw2 = eps * (v2 - gw2)
        # k3
        var v3 = v + hd1 * dv2
        var w3 = w + hd1 * dw2
        var dv3 = _fv(v3, half_a, mid, a) - w3 + current
        var gw3 = gamma * w3
        var dw3 = eps * (v3 - gw3)
        # k4
        var v4 = v + dt * dv3
        var w4 = w + dt * dw3
        var dv4 = _fv(v4, half_a, mid, a) - w4 + current
        var gw4 = gamma * w4
        var dw4 = eps * (v4 - gw4)
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
