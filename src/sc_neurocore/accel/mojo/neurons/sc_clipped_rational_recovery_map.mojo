# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained clipped rational-recovery map

from std.memory import UnsafePointer
from std.math import isfinite


@no_inline
def _rounded_product(lhs: Float64, rhs: Float64) -> Float64:
    """Round a product before the caller performs its next operation."""
    return lhs * rhs


@export
def sc_clipped_rational_recovery_map_simulate_c(
    x0: Float64,
    y0: Float64,
    alpha: Float64,
    beta: Float64,
    j: Float64,
    x_threshold: Float64,
    clip_bound: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if n_steps < 0 or trace_addr == 0:
        return -1
    if not (
        isfinite(x0)
        and isfinite(y0)
        and isfinite(alpha)
        and isfinite(beta)
        and isfinite(j)
        and isfinite(x_threshold)
        and isfinite(clip_bound)
        and isfinite(current)
    ):
        return -1
    if not (
        alpha > 0.0
        and beta > 0.0
        and clip_bound > 0.0
        and abs(x0) <= clip_bound
        and abs(y0) <= clip_bound
    ):
        return -1

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var events: Int64 = 0
    for index in range(n_steps):
        var x_previous = x
        var field = _rounded_product(alpha, x)
        if x >= 0.0:
            var numerator = _rounded_product(alpha, x)
            var denominator_term = _rounded_product(alpha, x)
            field = numerator / (1.0 + denominator_term)
        var x_candidate = field + y + current + j
        var recovery_term = _rounded_product(beta, x + 1.0)
        var y_candidate = y - recovery_term
        if not (isfinite(x_candidate) and isfinite(y_candidate)):
            return -1
        x = min(clip_bound, max(-clip_bound, x_candidate))
        y = min(clip_bound, max(-clip_bound, y_candidate))
        trace[index] = x
        if x >= x_threshold and x_previous < x_threshold:
            events += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return events
