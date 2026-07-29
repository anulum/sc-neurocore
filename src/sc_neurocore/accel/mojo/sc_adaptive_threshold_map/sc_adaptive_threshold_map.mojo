# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for the retained SC adaptive-threshold map

# Public contract: complete x/theta/event traces, bounded state, atomic failure.
# Build: mojo build --emit shared-lib -o libsc_adaptive_threshold_map.so sc_adaptive_threshold_map.mojo

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _sigmoid(value: Float64) -> Float64:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    var exponential = exp(value)
    return exponential / (1.0 + exponential)


@always_inline
def _clamp(value: Float64, lower: Float64, upper: Float64) -> Float64:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _run(
    n: Int32,
    x_init: Float64,
    theta_init: Float64,
    k: Float64,
    beta: Float64,
    gamma: Float64,
    theta_spike: Float64,
    x_threshold: Float64,
    current_addr: Int,
    x_out_addr: Int,
    theta_out_addr: Int,
    spikes_out_addr: Int,
    x_final_addr: Int,
    theta_final_addr: Int,
    spike_count_addr: Int,
    write_output: Bool,
) -> Int32:
    if n < 0 or x_final_addr == 0 or theta_final_addr == 0 or spike_count_addr == 0:
        return 1
    var steps = Int(n)
    if steps > 0 and (
        current_addr == 0 or x_out_addr == 0 or theta_out_addr == 0 or spikes_out_addr == 0
    ):
        return 1
    if (
        not isfinite(x_init) or x_init < -5.0 or x_init > 5.0
        or not isfinite(theta_init) or theta_init < -5.0 or theta_init > 5.0
        or not isfinite(k) or k < 0.0 or k > 5.0
        or not isfinite(beta) or beta < 0.0 or beta > 1.0
        or not isfinite(gamma) or gamma < 0.0 or gamma > 2.0
        or not isfinite(theta_spike) or theta_spike < 0.0 or theta_spike > 2.0
        or not isfinite(x_threshold) or x_threshold < 0.0 or x_threshold > 2.0
    ):
        return 2

    var x_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_final_addr)
    var theta_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta_final_addr)
    var count_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spike_count_addr)
    if steps == 0:
        if write_output:
            x_final[0] = x_init
            theta_final[0] = theta_init
            count_out[0] = 0.0
        return 0

    var current = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=current_addr)
    var x_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_out_addr)
    var theta_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta_out_addr)
    var spikes_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=spikes_out_addr)
    for index in range(steps):
        if not isfinite(current[index]):
            return 3

    var x = x_init
    var theta = theta_init
    var count = 0
    for index in range(steps):
        var previous_x = x
        var next_x = -x + k * _sigmoid((x - theta) * 4.0) + current[index]
        var fired = 0.0
        if x >= theta_spike:
            fired = 1.0
        var next_theta = beta * theta + gamma * fired
        if not isfinite(next_x) or not isfinite(next_theta):
            return 4
        x = _clamp(next_x, -5.0, 5.0)
        theta = _clamp(next_theta, -5.0, 5.0)
        var event = 0.0
        if x >= x_threshold and previous_x < x_threshold:
            event = 1.0
            count += 1
        if write_output:
            x_out[index] = x
            theta_out[index] = theta
            spikes_out[index] = event
    if write_output:
        x_final[0] = x
        theta_final[0] = theta
        count_out[0] = Float64(count)
    return 0


# Exported C ABI for the retained project-model batch.
@export
def sc_adaptive_threshold_map_simulate_c(
    n: Int32,
    x_init: Float64,
    theta_init: Float64,
    k: Float64,
    beta: Float64,
    gamma: Float64,
    theta_spike: Float64,
    x_threshold: Float64,
    current_addr: Int,
    x_out_addr: Int,
    theta_out_addr: Int,
    spikes_out_addr: Int,
    x_final_addr: Int,
    theta_final_addr: Int,
    spike_count_addr: Int,
) -> Int32:
    var status = _run(
        n, x_init, theta_init, k, beta, gamma, theta_spike, x_threshold,
        current_addr, x_out_addr, theta_out_addr, spikes_out_addr,
        x_final_addr, theta_final_addr, spike_count_addr, False,
    )
    if status != 0:
        return status
    return _run(
        n, x_init, theta_init, k, beta, gamma, theta_spike, x_threshold,
        current_addr, x_out_addr, theta_out_addr, spikes_out_addr,
        x_final_addr, theta_final_addr, spike_count_addr, True,
    )
