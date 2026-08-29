# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Ibarz analysis profile of the Shilnikov-Rulkov map

from std.memory import UnsafePointer
from std.math import isfinite


@export
def ibarz_tanaka_map_simulate_c(
    v0: Float64,
    u0: Float64,
    alpha: Float64,
    mu: Float64,
    sigma: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(v0)
        or not isfinite(u0)
        or not isfinite(alpha)
        or alpha <= 0.0
        or not isfinite(mu)
        or mu <= 0.0
        or not isfinite(sigma)
        or not isfinite(current)
    ):
        return -1

    # Validate the complete orbit before reconstructing or mutating the caller's
    # buffer. This preserves the C-ABI failure-atomic contract without a second
    # allocation in the accelerator runtime.
    var probe_v = v0
    var probe_u = u0
    for _step in range(n_steps):
        var probe_lower = -1.0 - alpha / 2.0
        var probe_upper = 1.0 + current + probe_u
        var probe_v_next: Float64
        if probe_v < probe_lower:
            var probe_alpha_squared = alpha * alpha
            probe_v_next = (
                -probe_alpha_squared / 4.0 - alpha + current + probe_u
            )
        elif probe_v <= 0.0:
            var probe_shifted = probe_v + 1.0
            var probe_linear = alpha * probe_v
            var probe_square = probe_shifted * probe_shifted
            probe_v_next = (
                probe_linear + probe_square + current + probe_u
            )
        elif probe_v < probe_upper:
            probe_v_next = probe_upper
        else:
            probe_v_next = -1.0
        var probe_slow_offset = probe_v + 1.0 - sigma
        var probe_slow_delta = mu * probe_slow_offset
        var probe_u_next = probe_u - probe_slow_delta
        if not isfinite(probe_v_next) or not isfinite(probe_u_next):
            return -2
        probe_v = probe_v_next
        probe_u = probe_u_next

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var u = u0
    var events: Int64 = 0
    for step in range(n_steps):
        var lower = -1.0 - alpha / 2.0
        var upper = 1.0 + current + u
        var v_next: Float64
        if v < lower:
            var alpha_squared = alpha * alpha
            v_next = -alpha_squared / 4.0 - alpha + current + u
        elif v <= 0.0:
            var shifted = v + 1.0
            var linear = alpha * v
            var square = shifted * shifted
            v_next = linear + square + current + u
        elif v < upper:
            v_next = upper
        else:
            v_next = -1.0
            events += 1
        var slow_offset = v + 1.0 - sigma
        var slow_delta = mu * slow_offset
        var u_next = u - slow_delta
        if not isfinite(v_next) or not isfinite(u_next):
            return -2  # unreachable after the identical validation pass
        v = v_next
        u = u_next
        trace[step] = v
    trace[n_steps] = v
    trace[n_steps + 1] = u
    return events
