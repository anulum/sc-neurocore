# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Rulkov 2001 fast/slow map (parity with rulkov_map.py)
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

from std.memory import UnsafePointer


@export
def rulkov_map_simulate_c(
    x0: Float64,
    y0: Float64,
    alpha: Float64,
    sigma: Float64,
    mu: Float64,
    x_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var x_prev = x
        var branch_boundary = alpha + y + current
        var x_new: Float64
        if x <= 0.0:
            x_new = alpha / (1.0 - x) + y + current
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
        x = x_new
        y = y_new
        trace[t] = x
        if x >= x_threshold and x_prev < x_threshold:
            spikes += 1
    if n_steps > 0:
        trace[n_steps] = x
        trace[n_steps + 1] = y
    return spikes
