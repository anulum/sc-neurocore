# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained upward-crossing Rulkov-map Mojo backend
#
# Build: mojo build --emit shared-lib -o libsc_upward_crossing_rulkov_map.so sc_upward_crossing_rulkov_map.mojo

from std.math import isfinite
from std.memory import UnsafePointer


@export
def sc_upward_crossing_rulkov_map_simulate_c(
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
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(x0)
        or not isfinite(y0)
        or not isfinite(alpha)
        or not isfinite(sigma)
        or not isfinite(mu)
        or not isfinite(x_threshold)
        or not isfinite(current)
        or alpha <= 0.0
        or mu <= 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var events: Int64 = 0
    for step in range(n_steps):
        var previous_x = x
        var boundary = alpha + y + current
        if not isfinite(boundary):
            return -1
        var x_next: Float64
        if x <= 0.0:
            var denominator = 1.0 - x
            if not isfinite(denominator) or denominator <= 0.0:
                return -1
            x_next = alpha / denominator + y + current
        elif x < boundary:
            x_next = boundary
        else:
            x_next = -1.0
        var shifted_x = x + 1.0
        var slow_x = mu * shifted_x
        var slow_sigma = mu * sigma
        var y_next = y - slow_x + slow_sigma
        if not isfinite(x_next) or not isfinite(y_next):
            return -1
        x = x_next
        y = y_next
        trace[step] = x
        if x >= x_threshold and previous_x < x_threshold:
            events += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return events
