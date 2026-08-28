# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Rulkov 2002 fast/slow map (parity with rulkov_map.py)
#
# Build:
#   mojo build --emit shared-lib -o librulkov.so rulkov_map.mojo
#
# Parity contract: `rulkov_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.rulkov_map.RulkovMapNeuron.simulate`. The fast
# map is exact floating-point arithmetic (one division, additions,
# multiplications, no transcendental functions). Rust/Julia/Go reproduce the
# trace bit-for-bit; Mojo's release build can contract the slow-variable update
# `y - mu*(x+1) + mu*sigma` into fused multiply-adds (one rounding instead of
# two), so each step agrees to within a couple of ULP. The branch resets
# (x -> exactly -1, x -> the plateau value) periodically resynchronise the
# trajectory, but the per-step ULP gap is real and matches the documented Mojo
# FMA-parity precedent for wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and
# the pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the x trace, index n the final x, index n+1 the final y.
#
# Reference: Rulkov, N.F. (2002). Phys. Rev. E 65:041922.

from std.math import isfinite
from std.memory import UnsafePointer


@export
def rulkov_map_simulate_c(
    x0: Float64,
    y0: Float64,
    alpha: Float64,
    sigma: Float64,
    mu: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(x0)
        or not isfinite(y0)
        or not isfinite(alpha)
        or not isfinite(sigma)
        or not isfinite(mu)
        or not isfinite(current)
        or alpha <= 0.0
        or mu <= 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var events: Int64 = 0
    for t in range(n_steps):
        var branch_boundary = alpha + y + current
        if not isfinite(branch_boundary):
            return -1
        var reset_event = x > 0.0 and x >= branch_boundary
        var x_new: Float64
        if x <= 0.0:
            var denominator = 1.0 - x
            if not isfinite(denominator) or denominator <= 0.0:
                return -1
            x_new = alpha / denominator + y + current
        elif x < branch_boundary:
            x_new = branch_boundary
        else:
            x_new = -1.0
        # Each product is rounded into its own variable before the surrounding
        # add/subtract so the compiler cannot contract `y - mu*(x+1) + mu*sigma`
        # into single-rounding fused multiply-adds — that fusion is the one
        # operation that diverges from the IEEE-754 two-rounding path used by
        # Python/Rust/Go/Julia.
        var x_plus_one = x + 1.0
        var mu_term = mu * x_plus_one
        var mu_sigma = mu * sigma
        var y_new = y - mu_term + mu_sigma
        if not isfinite(x_new) or not isfinite(y_new):
            return -1
        x = x_new
        y = y_new
        trace[t] = x
        if reset_event:
            events += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return events
