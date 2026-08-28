# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained clipped-logistic 2001 bursting map (parity with sc_clipped_logistic_bursting_map.py)
#
# Build:
#   mojo build --emit shared-lib -o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.mojo
#
# Parity contract: `sc_clipped_logistic_bursting_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.sc_clipped_logistic_bursting_map.retained clipped-logisticMapNeuron.simulate`
# bit-for-bit. The map is exact floating-point arithmetic (a*x*(1-x),
# additions, a clamp), so an identical operation order yields an identical
# trace, spike count, and final state.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and
# the pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the x trace, index n the final x, index n+1 the final y.
#
# Project-defined recurrence retained without whole-model attribution.

from std.memory import UnsafePointer


@always_inline
def _clamp_unit(v: Float64) -> Float64:
    if v < -2.0:
        return -2.0
    if v > 2.0:
        return 2.0
    return v


@export
def sc_clipped_logistic_bursting_map_simulate_c(
    x0: Float64,
    y0: Float64,
    a: Float64,
    epsilon: Float64,
    sigma: Float64,
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
        # Each product is rounded into its own variable before the following
        # add/subtract so the compiler cannot contract `y + epsilon*(x-sigma)`
        # (or the logistic term) into a single-rounding fused multiply-add —
        # that fusion is the one operation that diverges from the IEEE-754
        # two-rounding path used by Python/Rust/Go/Julia, and the chaotic map
        # amplifies a single ULP into a visible trace gap.
        var ax = a * x
        var one_minus_x = 1.0 - x
        var f = ax * one_minus_x
        var x_minus_sigma = x - sigma
        var eps_term = epsilon * x_minus_sigma
        var x_new = f - y + current
        var y_new = y + eps_term
        x = _clamp_unit(x_new)
        y = y_new
        trace[t] = x
        if x >= x_threshold:
            spikes += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return spikes
