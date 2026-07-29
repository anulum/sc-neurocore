# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for source-faithful Nagumo–Sato dynamics

# Public contract: complete y/output/event traces and atomic failure.
# Build: mojo build --emit shared-lib -o libnagumo_sato_map.so nagumo_sato_map.mojo

from std.math import isfinite
from std.memory import UnsafePointer


@always_inline
def _heaviside(value: Float64) -> Float64:
    if value >= 0.0:
        return 1.0
    return 0.0


def _run(
    n: Int32,
    y_init: Float64,
    k: Float64,
    alpha: Float64,
    bias: Float64,
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
    if (
        not isfinite(y_init) or not isfinite(k) or k < 0.0 or k >= 1.0
        or not isfinite(alpha) or alpha <= 0.0 or not isfinite(bias)
    ):
        return 2

    var y_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y_final_addr)
    var x_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_final_addr)
    var count_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spike_count_addr)
    if steps == 0:
        if write_output:
            y_final[0] = y_init
            x_final[0] = _heaviside(y_init)
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
        var next_y = k * y - alpha * _heaviside(y) + bias + current[index]
        if not isfinite(next_y):
            return 4
        y = next_y
        var event = _heaviside(y)
        if event > 0.0:
            count += 1
        if write_output:
            y_out[index] = y
            x_out[index] = event
            spikes_out[index] = event
    if write_output:
        y_final[0] = y
        x_final[0] = _heaviside(y)
        count_out[0] = Float64(count)
    return 0


# Exported C ABI for the source-faithful map batch.
@export
def nagumo_sato_map_simulate_c(
    n: Int32,
    y_init: Float64,
    k: Float64,
    alpha: Float64,
    bias: Float64,
    current_addr: Int,
    y_out_addr: Int,
    x_out_addr: Int,
    spikes_out_addr: Int,
    y_final_addr: Int,
    x_final_addr: Int,
    spike_count_addr: Int,
) -> Int32:
    var status = _run(
        n, y_init, k, alpha, bias, current_addr, y_out_addr, x_out_addr,
        spikes_out_addr, y_final_addr, x_final_addr, spike_count_addr, False,
    )
    if status != 0:
        return status
    return _run(
        n, y_init, k, alpha, bias, current_addr, y_out_addr, x_out_addr,
        spikes_out_addr, y_final_addr, x_final_addr, spike_count_addr, True,
    )
