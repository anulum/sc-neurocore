# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for source-faithful Aihara dynamics

# Build: mojo build --emit shared-lib -o libaihara_map.so aihara_map.mojo

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _logistic(value: Float64, epsilon: Float64) -> Float64:
    var argument = value / epsilon
    if argument >= 0.0:
        return 1.0 / (1.0 + exp(-argument))
    var exponential = exp(argument)
    return exponential / (1.0 + exponential)


@always_inline
def _ranges_overlap(
    a_addr: Int,
    a_elements: Int,
    b_addr: Int,
    b_elements: Int,
) -> Bool:
    """Compare two nonempty Float64 ranges without reading caller storage."""
    var a_bytes = a_elements * 8
    var b_bytes = b_elements * 8
    if a_addr <= b_addr:
        return b_addr - a_addr < a_bytes
    return a_addr - b_addr < b_bytes


@always_inline
def _active_regions_overlap(
    steps: Int,
    current_addr: Int,
    y_out_addr: Int,
    x_out_addr: Int,
    spikes_out_addr: Int,
    y_final_addr: Int,
    x_final_addr: Int,
    spike_count_addr: Int,
) -> Bool:
    """Detect any overlap among active input, trace and scalar output ranges."""
    if (
        _ranges_overlap(y_final_addr, 1, x_final_addr, 1)
        or _ranges_overlap(y_final_addr, 1, spike_count_addr, 1)
        or _ranges_overlap(x_final_addr, 1, spike_count_addr, 1)
    ):
        return True
    if steps == 0:
        return False
    return (
        _ranges_overlap(current_addr, steps, y_out_addr, steps)
        or _ranges_overlap(current_addr, steps, x_out_addr, steps)
        or _ranges_overlap(current_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(current_addr, steps, y_final_addr, 1)
        or _ranges_overlap(current_addr, steps, x_final_addr, 1)
        or _ranges_overlap(current_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(y_out_addr, steps, x_out_addr, steps)
        or _ranges_overlap(y_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(y_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(y_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(y_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(x_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(x_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(x_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(x_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, spike_count_addr, 1)
    )


def _run(
    n: Int32,
    y_init: Float64,
    k: Float64,
    alpha: Float64,
    bias: Float64,
    epsilon: Float64,
    current_addr: Int,
    y_out_addr: Int,
    x_out_addr: Int,
    spikes_out_addr: Int,
    y_final_addr: Int,
    x_final_addr: Int,
    spike_count_addr: Int,
    write_output: Bool,
) -> Int32:
    if n < 0 or y_final_addr == 0 or x_final_addr == 0 or spike_count_addr == 0:
        return 1
    var steps = Int(n)
    if steps > 0 and (
        current_addr == 0 or y_out_addr == 0 or x_out_addr == 0 or spikes_out_addr == 0
    ):
        return 1
    if _active_regions_overlap(
        steps, current_addr, y_out_addr, x_out_addr, spikes_out_addr,
        y_final_addr, x_final_addr, spike_count_addr,
    ):
        return 1
    if (
        not isfinite(y_init) or not isfinite(k) or k < 0.0 or k >= 1.0
        or not isfinite(alpha) or alpha <= 0.0 or not isfinite(bias)
        or not isfinite(epsilon) or epsilon <= 0.0
    ):
        return 2

    var y_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y_final_addr)
    var x_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_final_addr)
    var count_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spike_count_addr)
    if steps == 0:
        if write_output:
            y_final[0] = y_init
            x_final[0] = _logistic(y_init, epsilon)
            count_out[0] = 0.0
        return 0

    var current = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=current_addr)
    var y_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y_out_addr)
    var x_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_out_addr)
    var spikes_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spikes_out_addr)
    for index in range(steps):
        if not isfinite(current[index]):
            return 3

    var y = y_init
    var count = 0
    for index in range(steps):
        var next_y = k * y - alpha * _logistic(y, epsilon) + bias + current[index]
        if not isfinite(next_y):
            return 4
        y = next_y
        var x = _logistic(y, epsilon)
        var event = 0.0
        if x >= 0.5:
            event = 1.0
            count += 1
        if write_output:
            y_out[index] = y
            x_out[index] = x
            spikes_out[index] = event
    if write_output:
        y_final[0] = y
        x_final[0] = _logistic(y, epsilon)
        count_out[0] = Float64(count)
    return 0


@export
def aihara_map_simulate_c(
    n: Int32,
    y_init: Float64,
    k: Float64,
    alpha: Float64,
    bias: Float64,
    epsilon: Float64,
    current_addr: Int,
    y_out_addr: Int,
    x_out_addr: Int,
    spikes_out_addr: Int,
    y_final_addr: Int,
    x_final_addr: Int,
    spike_count_addr: Int,
) -> Int32:
    """Write the full source-map receipt into disjoint caller-owned buffers.

    Four Float64 ranges have n elements; three final outputs have one each.
    Empty traces may have null addresses. Status 1 rejects null/overlapping
    active buffers, 2 invalid configuration, 3 nonfinite input, 4 overflow.
    Rejection never writes caller storage. Allocations and their lifetimes
    remain the caller's responsibility; addresses are not allocation proofs.
    """
    var status = _run(
        n, y_init, k, alpha, bias, epsilon, current_addr, y_out_addr,
        x_out_addr, spikes_out_addr, y_final_addr, x_final_addr,
        spike_count_addr, False,
    )
    if status != 0:
        return status
    return _run(
        n, y_init, k, alpha, bias, epsilon, current_addr, y_out_addr,
        x_out_addr, spikes_out_addr, y_final_addr, x_final_addr,
        spike_count_addr, True,
    )
