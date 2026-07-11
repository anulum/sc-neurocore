# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI accelerator for the Chialvo map

# Build from this directory:
#   mojo build --emit shared-lib -o libchialvo.so chialvo_map.mojo
#
# The caller supplies n_steps+2 Float64 slots: the x trace, final x, final y.
# A negative return value reports rejected input or a non-finite candidate.

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _safe_exp_chialvo(value: Float64) -> Float64:
    var bounded = value
    if bounded < -500.0:
        bounded = -500.0
    elif bounded > 500.0:
        bounded = 500.0
    return exp(bounded)


@export
def chialvo_map_simulate_c(
    x0: Float64,
    y0: Float64,
    a: Float64,
    b: Float64,
    c: Float64,
    k: Float64,
    x_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if n_steps < 0 or trace_addr == 0:
        return -1
    if (
        not isfinite(x0)
        or not isfinite(y0)
        or not isfinite(a)
        or not isfinite(b)
        or not isfinite(c)
        or not isfinite(k)
        or not isfinite(x_threshold)
        or not isfinite(current)
    ):
        return -1

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var y = y0
    var spikes: Int64 = 0
    for index in range(n_steps):
        var x_previous = x
        var x_squared = x * x
        var exponential = _safe_exp_chialvo(y - x)
        var x_next = x_squared * exponential + k + current
        var y_next = a * y - b * x + c
        if not isfinite(x_next) or not isfinite(y_next):
            return -1
        x = x_next
        y = y_next
        trace[index] = x
        if x_previous < x_threshold and x >= x_threshold:
            spikes += 1
    trace[n_steps] = x
    trace[n_steps + 1] = y
    return spikes
