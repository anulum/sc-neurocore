# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Ibarz-Tanaka piecewise-linear map (parity with ibarz_tanaka_map.py)
#
# Build:
#   mojo build --emit shared-lib -o libibarz.so ibarz_tanaka_map.mojo
#
# Parity contract: `ibarz_tanaka_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron.simulate`.
# The map is exact floating-point arithmetic (one division, additions,
# multiplications, no transcendental functions). Rust/Julia/Go reproduce the
# trace bit-for-bit; Mojo's release build can contract the slow-variable update
# `y - mu*(x+1) + mu*sigma` (and the linear branch `alpha + beta*x`) into fused
# multiply-adds (one rounding instead of two), so each step agrees to within a
# couple of ULP. The explicit reset to `x_reset` on every spike periodically
# resynchronises the trajectory, so the whole-trace gap stays at the per-step
# ULP level — matching the documented Mojo FMA-parity precedent for
# wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and
# the pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the x trace, index n the final x, index n+1 the final y.
#
# Reference: Ibarz, B., Casado, J.M. & Sanjuán, M.A.F. (2011). Phys. Rep. 501:1-74.

from std.memory import UnsafePointer


@export
fn ibarz_tanaka_map_simulate_c(
    x0: Float64,
    y0: Float64,
    alpha: Float64,
    beta: Float64,
    mu: Float64,
    sigma: Float64,
    x_threshold: Float64,
    x_reset: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var f: Float64
        if x <= 0.0:
            f = alpha / (1.0 - x)
        else:
            # Round the product before the add so the compiler cannot fuse
            # `alpha + beta*x` into a single-rounding multiply-add.
            var beta_x = beta * x
            f = alpha + beta_x
        var x_new = f + y + current
        # Each product is rounded into its own variable before the surrounding
        # add/subtract so the compiler cannot contract `y - mu*(x+1) + mu*sigma`
        # into single-rounding fused multiply-adds — that fusion is the one
        # operation that diverges from the IEEE-754 two-rounding path used by
        # Python/Rust/Go/Julia.
        var x_plus_one = x + 1.0
        var mu_term = mu * x_plus_one
        var mu_sigma = mu * sigma
        var y_new = y - mu_term + mu_sigma
        y = y_new
        if x_new >= x_threshold:
            x = x_reset
            spikes += 1
        else:
            x = x_new
        trace[t] = x
    if n_steps > 0:
        trace[n_steps] = x
        trace[n_steps + 1] = y
    return spikes
