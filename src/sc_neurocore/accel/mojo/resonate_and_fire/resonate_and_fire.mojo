# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic Mojo batch mirror for Izhikevich resonate-and-fire
#
# Build: mojo build --Werror --target-cpu x86-64-v3 --emit shared-lib -o libresonate_and_fire.so resonate_and_fire.mojo

from std.math import cos, exp, isfinite, sin
from std.memory import Pointer


@always_inline
def _valid_configuration(
    x: Float64,
    y: Float64,
    b: Float64,
    omega: Float64,
    threshold: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(x)
        and isfinite(y)
        and isfinite(b)
        and isfinite(omega)
        and omega > 0.0
        and isfinite(threshold)
        and threshold > 0.0
        and isfinite(dt)
        and dt > 0.0
    )


@always_inline
def _ranges_overlap(
    a_addr: Int,
    a_elements: Int,
    b_addr: Int,
    b_elements: Int,
) -> Bool:
    var a_bytes = a_elements * 8
    var b_bytes = b_elements * 8
    if a_addr <= b_addr:
        return b_addr - a_addr < a_bytes
    return a_addr - b_addr < b_bytes


@always_inline
def _active_regions_overlap(
    steps: Int,
    current_addr: Int,
    x_out_addr: Int,
    y_out_addr: Int,
    spikes_out_addr: Int,
    x_final_addr: Int,
    y_final_addr: Int,
    spike_count_addr: Int,
) -> Bool:
    if (
        _ranges_overlap(x_final_addr, 1, y_final_addr, 1)
        or _ranges_overlap(x_final_addr, 1, spike_count_addr, 1)
        or _ranges_overlap(y_final_addr, 1, spike_count_addr, 1)
    ):
        return True
    if steps == 0:
        return False
    return (
        _ranges_overlap(current_addr, steps, x_out_addr, steps)
        or _ranges_overlap(current_addr, steps, y_out_addr, steps)
        or _ranges_overlap(current_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(current_addr, steps, x_final_addr, 1)
        or _ranges_overlap(current_addr, steps, y_final_addr, 1)
        or _ranges_overlap(current_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(x_out_addr, steps, y_out_addr, steps)
        or _ranges_overlap(x_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(x_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(x_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(x_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(y_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(y_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(y_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(y_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, x_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, y_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, spike_count_addr, 1)
    )


@always_inline
def _exact_x(
    x: Float64,
    y: Float64,
    current: Float64,
    b: Float64,
    omega: Float64,
    dt: Float64,
) -> Float64:
    var denominator = b * b + omega * omega
    var x_ss = -b * current / denominator
    var y_ss = omega * current / denominator
    var decay = exp(b * dt)
    var angle = omega * dt
    return x_ss + decay * (
        (x - x_ss) * cos(angle) - (y - y_ss) * sin(angle)
    )


@always_inline
def _exact_y(
    x: Float64,
    y: Float64,
    current: Float64,
    b: Float64,
    omega: Float64,
    dt: Float64,
) -> Float64:
    var denominator = b * b + omega * omega
    var x_ss = -b * current / denominator
    var y_ss = omega * current / denominator
    var decay = exp(b * dt)
    var angle = omega * dt
    return y_ss + decay * (
        (x - x_ss) * sin(angle) + (y - y_ss) * cos(angle)
    )


def _run_resonate_and_fire(
    n: Int32,
    x_init: Float64,
    y_init: Float64,
    b: Float64,
    omega: Float64,
    threshold: Float64,
    dt: Float64,
    current_addr: Int,
    x_out_addr: Int,
    y_out_addr: Int,
    spikes_out_addr: Int,
    x_final_addr: Int,
    y_final_addr: Int,
    spike_count_addr: Int,
    write_output: Bool,
) -> Int32:
    if (
        n < 0
        or x_final_addr == 0
        or y_final_addr == 0
        or spike_count_addr == 0
    ):
        return 1
    var steps = Int(n)
    if steps > 0 and (
        current_addr == 0
        or x_out_addr == 0
        or y_out_addr == 0
        or spikes_out_addr == 0
    ):
        return 1
    if _active_regions_overlap(
        steps,
        current_addr,
        x_out_addr,
        y_out_addr,
        spikes_out_addr,
        x_final_addr,
        y_final_addr,
        spike_count_addr,
    ):
        return 1
    if not _valid_configuration(x_init, y_init, b, omega, threshold, dt):
        return 2

    var x_final = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=x_final_addr
    )
    var y_final = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=y_final_addr
    )
    var spike_count = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=spike_count_addr
    )
    if steps == 0:
        if write_output:
            x_final[unsafe_offset=0], y_final[unsafe_offset=0], spike_count[unsafe_offset=0] = x_init, y_init, 0.0
        return 0

    var current = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=current_addr
    )
    var x_out = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=x_out_addr
    )
    var y_out = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=y_out_addr
    )
    var spikes_out = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )
    for index in range(steps):
        if not isfinite(current[unsafe_offset=index]):
            return 3

    var x = x_init
    var y = y_init
    var count = 0
    for index in range(steps):
        var next_x = _exact_x(x, y, current[unsafe_offset=index], b, omega, dt)
        var next_y = _exact_y(x, y, current[unsafe_offset=index], b, omega, dt)
        if not isfinite(next_x) or not isfinite(next_y):
            return 4
        var spike = 0.0
        if y < threshold and next_y >= threshold:
            x, y = 0.0, threshold
            spike = 1.0
            count += 1
        else:
            x, y = next_x, next_y
        if write_output:
            x_out[unsafe_offset=index], y_out[unsafe_offset=index], spikes_out[unsafe_offset=index] = x, y, spike
    if write_output:
        x_final[unsafe_offset=0], y_final[unsafe_offset=0], spike_count[unsafe_offset=0] = x, y, Float64(count)
    return 0


# The caller owns stable Float64 current[n] and three disjoint writable
# Float64 traces[n] (x, y, binary events), plus disjoint Float64 final x, final y
# and event-count scalars. Buffers remain valid for this synchronous call.
# Zero steps allow null trace addresses but require the three final-state scalars.
# Status 0 writes the complete result; 1 rejects lengths, addresses or overlap,
# 2 invalid configuration, 3 nonfinite current, 4 nonfinite dynamics.
# Every refusal leaves outputs unchanged; no pointer ownership is retained.
@export
def resonate_and_fire_simulate_c(
    n: Int32,
    x_init: Float64,
    y_init: Float64,
    b: Float64,
    omega: Float64,
    threshold: Float64,
    dt: Float64,
    current_addr: Int,
    x_out_addr: Int,
    y_out_addr: Int,
    spikes_out_addr: Int,
    x_final_addr: Int,
    y_final_addr: Int,
    spike_count_addr: Int,
) abi("C") -> Int32:
    var status = _run_resonate_and_fire(
        n,
        x_init,
        y_init,
        b,
        omega,
        threshold,
        dt,
        current_addr,
        x_out_addr,
        y_out_addr,
        spikes_out_addr,
        x_final_addr,
        y_final_addr,
        spike_count_addr,
        False,
    )
    if status != 0:
        return status
    return _run_resonate_and_fire(
        n,
        x_init,
        y_init,
        b,
        omega,
        threshold,
        dt,
        current_addr,
        x_out_addr,
        y_out_addr,
        spikes_out_addr,
        x_final_addr,
        y_final_addr,
        spike_count_addr,
        True,
    )
