# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for the SC two-state chaotic map

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _logistic(value: Float64) -> Float64:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    var exponential = exp(value)
    return exponential / (1.0 + exponential)


def _run(
    n: Int32, x_init: Float64, y_init: Float64, k_f: Float64, k_s: Float64,
    alpha: Float64, delta: Float64, threshold: Float64, current_addr: Int,
    x_out_addr: Int, y_out_addr: Int, spikes_out_addr: Int, x_final_addr: Int,
    y_final_addr: Int, spike_count_addr: Int, write_output: Bool,
) -> Int32:
    if n < 0 or x_final_addr == 0 or y_final_addr == 0 or spike_count_addr == 0:
        return 1
    var steps = Int(n)
    if steps > 0 and (
        current_addr == 0 or x_out_addr == 0 or y_out_addr == 0 or spikes_out_addr == 0
    ):
        return 1
    if (
        not isfinite(x_init) or not isfinite(y_init) or not isfinite(k_f) or k_f < 0.0
        or not isfinite(k_s) or not isfinite(alpha) or not isfinite(delta) or delta < 0.0
        or not isfinite(threshold)
    ):
        return 2
    var x_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_final_addr)
    var y_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y_final_addr)
    var count_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spike_count_addr)
    if steps == 0:
        if write_output:
            x_final[0] = x_init
            y_final[0] = y_init
            count_out[0] = 0.0
        return 0
    var current = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=current_addr)
    var x_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_out_addr)
    var y_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y_out_addr)
    var spikes_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spikes_out_addr)
    for index in range(steps):
        if not isfinite(current[index]):
            return 3
    var x = x_init
    var y = y_init
    var count = 0
    for index in range(steps):
        var previous = x
        var next_x = k_f * x * _logistic(x + alpha) - y + current[index]
        var next_y = k_s * y + delta * x
        if not isfinite(next_x) or not isfinite(next_y):
            return 4
        x = max(-10.0, min(10.0, next_x))
        y = max(-10.0, min(10.0, next_y))
        var event = 0.0
        if previous < threshold and x >= threshold:
            event = 1.0
            count += 1
        if write_output:
            x_out[index] = x
            y_out[index] = y
            spikes_out[index] = event
    if write_output:
        x_final[0] = x
        y_final[0] = y
        count_out[0] = Float64(count)
    return 0


@export
def sc_chaotic_map_simulate_c(
    n: Int32, x_init: Float64, y_init: Float64, k_f: Float64, k_s: Float64,
    alpha: Float64, delta: Float64, threshold: Float64, current_addr: Int,
    x_out_addr: Int, y_out_addr: Int, spikes_out_addr: Int, x_final_addr: Int,
    y_final_addr: Int, spike_count_addr: Int,
) -> Int32:
    var status = _run(
        n, x_init, y_init, k_f, k_s, alpha, delta, threshold, current_addr,
        x_out_addr, y_out_addr, spikes_out_addr, x_final_addr, y_final_addr,
        spike_count_addr, False,
    )
    if status != 0:
        return status
    return _run(
        n, x_init, y_init, k_f, k_s, alpha, delta, threshold, current_addr,
        x_out_addr, y_out_addr, spikes_out_addr, x_final_addr, y_final_addr,
        spike_count_addr, True,
    )
