# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Cazelles four-branch map

from std.math import isfinite
from std.memory import UnsafePointer


@export
def cazelles_map_simulate_c(
    x0_state: Float64,
    alpha: Float64,
    x0: Float64,
    x1: Float64,
    x2: Float64,
    x3: Float64,
    x4: Float64,
    a1: Float64,
    a2: Float64,
    a3: Float64,
    a4: Float64,
    b1: Float64,
    b2: Float64,
    b3: Float64,
    b4: Float64,
    exponent: Int,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(x0_state)
        or not isfinite(alpha)
        or not isfinite(x0)
        or not isfinite(x1)
        or not isfinite(x2)
        or not isfinite(x3)
        or not isfinite(x4)
        or not isfinite(a1)
        or not isfinite(a2)
        or not isfinite(a3)
        or not isfinite(a4)
        or not isfinite(b1)
        or not isfinite(b2)
        or not isfinite(b3)
        or not isfinite(b4)
        or not isfinite(current)
        or alpha < 0.0
        or alpha >= 1.0
        or (exponent != 1 and exponent != 2)
        or not (x0 < x1 and x1 < x2 and x2 < x3 and x3 < x4)
        or x0_state < x0
        or x0_state > x4
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0_state
    var events: Int64 = 0
    for index in range(n_steps):
        var base: Float64
        if x < x1:
            base = a1 + b1 * x
        elif x < x2:
            base = a2 + b2 * x
        elif x < x3:
            base = a3 + b3 * x
        else:
            base = a4 + b4 * x
        var power = x
        if exponent == 2:
            power = x * x
        var alpha_term = alpha * power
        var candidate = base + alpha_term + current
        var scale = max(1.0, max(abs(x0), abs(x4)))
        var tolerance = 8.0 * 2.220446049250313e-16 * scale
        if candidate < x0 and candidate >= x0 - tolerance:
            candidate = x0
        elif candidate > x4 and candidate <= x4 + tolerance:
            candidate = x4
        if not isfinite(candidate) or candidate < x0 or candidate > x4:
            trace[n_steps] = x
            return -2
        if x >= x1 and candidate < x1:
            events += 1
        x = candidate
        trace[index] = x
    trace[n_steps] = x
    return events
