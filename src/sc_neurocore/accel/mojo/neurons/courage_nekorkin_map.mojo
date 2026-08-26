# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Courbage-Nekorkin-Vdovin 2007 map (parity with courage_nekorkin_map.py)
#
# Build:
#   mojo build --emit shared-lib -o libcourage.so courage_nekorkin_map.mojo
#
# Parity contract: `courage_nekorkin_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.courage_nekorkin_map.CourageNekorkinMapNeuron.simulate`.
# The map is chaotic exact-arithmetic dynamics: each product is rounded into its
# own variable before the following add/subtract so the compiler cannot contract
# a multiply-add into a single-rounding FMA — that fusion is the one operation
# that diverges from the IEEE-754 two-rounding path used by Python/Rust/Go/Julia.
# Because the map is chaotic any residual single-ULP difference still amplifies
# over a long trace, so the Mojo backend is validated per-step and on spike
# counts rather than on the whole trace.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the x trace, index n the final x, index n+1 the final y.
#
# Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
# Chaos 17:043109 (arXiv:0712.2097), eqs. 3-5.

from std.memory import UnsafePointer


@export
def courage_nekorkin_map_simulate_c(
    x0: Float64,
    y0: Float64,
    m0: Float64,
    m1: Float64,
    a: Float64,
    d: Float64,
    j: Float64,
    beta: Float64,
    eps: Float64,
    x_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var am1 = a * m1
    var den = m0 + m1
    var jmin = am1 / den
    var jmax = (m0 + am1) / den
    var spikes: Int64 = 0
    for t in range(n_steps):
        var x_prev = x
        var fx: Float64
        if x <= jmin:
            fx = -m0 * x
        elif x < jmax:
            var x_minus_a = x - a
            fx = m1 * x_minus_a
        else:
            var x_minus_one = x - 1.0
            fx = -m0 * x_minus_one
        var h: Float64 = 1.0 if (x - d) >= 0.0 else 0.0
        var beta_h = beta * h
        var x_minus_j = x - j
        var eps_term = eps * x_minus_j
        var x_new = x + fx - y - beta_h + current
        var y_new = y + eps_term
        x = x_new
        y = y_new
        trace[t] = x
        if x >= x_threshold and x_prev < x_threshold:
            spikes += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return spikes
