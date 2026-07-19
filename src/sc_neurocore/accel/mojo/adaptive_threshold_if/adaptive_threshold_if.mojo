# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic Mojo batch mirror for composite reduced adaptive-threshold IF

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _valid_configuration(
    v: Float64,
    theta: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(v)
        and isfinite(theta)
        and isfinite(v_rest)
        and isfinite(v_reset)
        and isfinite(theta_rest)
        and theta_rest > v_rest
        and theta_rest > v_reset
        and isfinite(delta_theta)
        and delta_theta >= 0.0
        and isfinite(tau_m)
        and tau_m > 0.0
        and isfinite(tau_theta)
        and tau_theta > 0.0
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
    v_out_addr: Int,
    theta_out_addr: Int,
    spikes_out_addr: Int,
    v_final_addr: Int,
    theta_final_addr: Int,
    spike_count_addr: Int,
) -> Bool:
    if (
        _ranges_overlap(v_final_addr, 1, theta_final_addr, 1)
        or _ranges_overlap(v_final_addr, 1, spike_count_addr, 1)
        or _ranges_overlap(theta_final_addr, 1, spike_count_addr, 1)
    ):
        return True
    if steps == 0:
        return False
    return (
        _ranges_overlap(current_addr, steps, v_out_addr, steps)
        or _ranges_overlap(current_addr, steps, theta_out_addr, steps)
        or _ranges_overlap(current_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(current_addr, steps, v_final_addr, 1)
        or _ranges_overlap(current_addr, steps, theta_final_addr, 1)
        or _ranges_overlap(current_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(v_out_addr, steps, theta_out_addr, steps)
        or _ranges_overlap(v_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(v_out_addr, steps, v_final_addr, 1)
        or _ranges_overlap(v_out_addr, steps, theta_final_addr, 1)
        or _ranges_overlap(v_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(theta_out_addr, steps, spikes_out_addr, steps)
        or _ranges_overlap(theta_out_addr, steps, v_final_addr, 1)
        or _ranges_overlap(theta_out_addr, steps, theta_final_addr, 1)
        or _ranges_overlap(theta_out_addr, steps, spike_count_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, v_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, theta_final_addr, 1)
        or _ranges_overlap(spikes_out_addr, steps, spike_count_addr, 1)
    )


@always_inline
def _exact_relaxation(
    state: Float64,
    steady_state: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    return steady_state + (state - steady_state) * exp(-dt / tau)


def _run_adaptive_threshold_if(
    n: Int32,
    v_init: Float64,
    theta_init: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    dt: Float64,
    current_addr: Int,
    v_out_addr: Int,
    theta_out_addr: Int,
    spikes_out_addr: Int,
    v_final_addr: Int,
    theta_final_addr: Int,
    spike_count_addr: Int,
    write_output: Bool,
) -> Int32:
    if (
        n < 0
        or v_final_addr == 0
        or theta_final_addr == 0
        or spike_count_addr == 0
    ):
        return 1
    var steps = Int(n)
    if steps > 0 and (
        current_addr == 0
        or v_out_addr == 0
        or theta_out_addr == 0
        or spikes_out_addr == 0
    ):
        return 1
    if _active_regions_overlap(
        steps,
        current_addr,
        v_out_addr,
        theta_out_addr,
        spikes_out_addr,
        v_final_addr,
        theta_final_addr,
        spike_count_addr,
    ):
        return 1
    if not _valid_configuration(
        v_init, theta_init, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt
    ):
        return 2

    var v_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_final_addr
    )
    var theta_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=theta_final_addr
    )
    var spike_count = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=spike_count_addr
    )
    if steps == 0:
        if write_output:
            v_final[0], theta_final[0], spike_count[0] = v_init, theta_init, 0.0
        return 0

    var current = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=current_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_out_addr
    )
    var theta_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=theta_out_addr
    )
    var spikes_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )
    for index in range(steps):
        if not isfinite(current[index]):
            return 3

    var v = v_init
    var theta = theta_init
    var count = 0
    for index in range(steps):
        var next_v = _exact_relaxation(v, v_rest + current[index], tau_m, dt)
        var next_theta = _exact_relaxation(theta, theta_rest, tau_theta, dt)
        if not isfinite(next_v) or not isfinite(next_theta):
            return 4
        var spike = 0.0
        if next_v >= next_theta:
            var spike_theta = next_theta + delta_theta
            if not isfinite(spike_theta):
                return 4
            v, theta = v_reset, spike_theta
            spike = 1.0
            count += 1
        else:
            v, theta = next_v, next_theta
        if write_output:
            v_out[index], theta_out[index], spikes_out[index] = v, theta, spike
    if write_output:
        v_final[0], theta_final[0], spike_count[0] = v, theta, Float64(count)
    return 0


@export
def adaptive_threshold_if_simulate_c(
    n: Int32,
    v_init: Float64,
    theta_init: Float64,
    v_rest: Float64,
    v_reset: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    dt: Float64,
    current_addr: Int,
    v_out_addr: Int,
    theta_out_addr: Int,
    spikes_out_addr: Int,
    v_final_addr: Int,
    theta_final_addr: Int,
    spike_count_addr: Int,
) -> Int32:
    var status = _run_adaptive_threshold_if(
        n,
        v_init,
        theta_init,
        v_rest,
        v_reset,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        dt,
        current_addr,
        v_out_addr,
        theta_out_addr,
        spikes_out_addr,
        v_final_addr,
        theta_final_addr,
        spike_count_addr,
        False,
    )
    if status != 0:
        return status
    return _run_adaptive_threshold_if(
        n,
        v_init,
        theta_init,
        v_rest,
        v_reset,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        dt,
        current_addr,
        v_out_addr,
        theta_out_addr,
        spikes_out_addr,
        v_final_addr,
        theta_final_addr,
        spike_count_addr,
        True,
    )
