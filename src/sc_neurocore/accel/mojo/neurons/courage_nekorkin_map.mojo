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
# The map is chaotic exact-arithmetic dynamics. `_rounded_product` is deliberately
# not inlined, preserving the IEEE-754 product rounding before each following
# addition or subtraction. This prevents FMA contraction and reproduces the
# Python/Rust/Go/Julia binary64 operation order over the complete trajectory.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the x trace, index n the final x, index n+1 the final y.
#
# Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
# Chaos 17:043109 (arXiv:0712.2097), eqs. 3-5.

from std.memory import UnsafePointer
from std.math import isfinite


@no_inline
def _rounded_product(lhs: Float64, rhs: Float64) -> Float64:
    """Round a product before the caller performs its next operation."""
    return lhs * rhs


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
    if n_steps < 0 or trace_addr == 0:
        return -1
    if not (
        isfinite(x0)
        and isfinite(y0)
        and isfinite(m0)
        and isfinite(m1)
        and isfinite(a)
        and isfinite(d)
        and isfinite(j)
        and isfinite(beta)
        and isfinite(eps)
        and isfinite(x_threshold)
        and isfinite(current)
    ):
        return -1
    if not (
        m0 > 0.0
        and m0 < 1.0
        and m1 > 0.0
        and a > 0.0
        and a < 1.0
        and d > 0.0
        and beta > 0.0
        and eps > 0.0
        and j > 0.0
        and j < d
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var am1 = a * m1
    var den = m0 + m1
    var jmin = am1 / den
    var jmax = (m0 + am1) / den
    if not (jmin < d and d < jmax):
        return -1
    var spikes: Int64 = 0
    for t in range(n_steps):
        var x_prev = x
        var fx: Float64
        if x <= jmin:
            fx = _rounded_product(-m0, x)
        elif x < jmax:
            var x_minus_a = x - a
            fx = _rounded_product(m1, x_minus_a)
        else:
            var x_minus_one = x - 1.0
            fx = _rounded_product(-m0, x_minus_one)
        var h: Float64 = 1.0 if (x - d) >= 0.0 else 0.0
        var beta_h = _rounded_product(beta, h)
        var x_minus_j = x - j
        var eps_term = _rounded_product(eps, x_minus_j)
        var x_new = x + fx - y - beta_h + current
        var y_new = y + eps_term
        if not (isfinite(x_new) and isfinite(y_new)):
            return -1
        x = x_new
        y = y_new
        trace[t] = x
        if x >= x_threshold and x_prev < x_threshold:
            spikes += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return spikes
