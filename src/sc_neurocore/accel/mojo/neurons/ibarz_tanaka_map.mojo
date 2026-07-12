# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Ibarz-Tanaka 2007 four-branch map

from std.memory import UnsafePointer


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
        v = v_next
        u = u_next
        trace[step] = v
    trace[n_steps] = v
    trace[n_steps + 1] = u
    return events
