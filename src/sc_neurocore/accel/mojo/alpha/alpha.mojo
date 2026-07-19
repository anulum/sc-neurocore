# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic Mojo batch mirror for dual alpha-synapse LIF

from std.math import abs, exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _valid_configuration(
    v: Float64,
    a_exc: Float64,
    i_exc: Float64,
    a_inh: Float64,
    i_inh: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_v: Float64,
    tau_exc: Float64,
    tau_inh: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(v)
        and isfinite(a_exc)
        and isfinite(i_exc)
        and isfinite(a_inh)
        and isfinite(i_inh)
        and isfinite(v_rest)
        and isfinite(v_threshold)
        and v_threshold > v_rest
        and isfinite(tau_v)
        and tau_v > 0.0
        and isfinite(tau_exc)
        and tau_exc > 0.0
        and isfinite(tau_inh)
        and tau_inh > 0.0
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
def _filter_next(
    rise_state: Float64,
    current_state: Float64,
    drive: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    var steady_state = tau * drive
    var rise_delta = rise_state - steady_state
    var current_delta = current_state - steady_state
    return steady_state + exp(-dt / tau) * (current_delta + rise_delta * dt / tau)


@always_inline
def _rise_next(
    rise_state: Float64,
    drive: Float64,
    tau: Float64,
    dt: Float64,
) -> Float64:
    var steady_state = tau * drive
    return steady_state + (rise_state - steady_state) * exp(-dt / tau)


@always_inline
def _drive_contribution(
    current_delta: Float64,
    rise_delta: Float64,
    tau_drive: Float64,
    tau_v: Float64,
    dt: Float64,
) -> Float64:
    var rate_v = 1.0 / tau_v
    var rate_drive = 1.0 / tau_drive
    var decay_v = exp(-dt / tau_v)
    var decay_drive = exp(-dt / tau_drive)
    if abs(rate_v - rate_drive) <= 1.0e-14:
        return rate_v * decay_v * (
            current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive)
        )
    var rate_delta = rate_v - rate_drive
    var first_order = current_delta * (decay_drive - decay_v) / rate_delta
    var second_order = rise_delta / tau_drive * (
        decay_drive * (rate_delta * dt - 1.0) + decay_v
    ) / (rate_delta * rate_delta)
    return rate_v * (first_order + second_order)


def _run_alpha(
    n: Int32,
    v_init: Float64,
    a_exc_init: Float64,
    i_exc_init: Float64,
    a_inh_init: Float64,
    i_inh_init: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_v: Float64,
    tau_exc: Float64,
    tau_inh: Float64,
    dt: Float64,
    exc_current_addr: Int,
    inh_current_addr: Int,
    v_out_addr: Int,
    a_exc_out_addr: Int,
    i_exc_out_addr: Int,
    a_inh_out_addr: Int,
    i_inh_out_addr: Int,
    spikes_out_addr: Int,
    v_final_addr: Int,
    a_exc_final_addr: Int,
    i_exc_final_addr: Int,
    a_inh_final_addr: Int,
    i_inh_final_addr: Int,
    spike_count_addr: Int,
    write_output: Bool,
) -> Int32:
    if (
        n < 0
        or v_final_addr == 0
        or a_exc_final_addr == 0
        or i_exc_final_addr == 0
        or a_inh_final_addr == 0
        or i_inh_final_addr == 0
        or spike_count_addr == 0
    ):
        return 1
    var steps = Int(n)
    if steps > 0 and (
        exc_current_addr == 0
        or inh_current_addr == 0
        or v_out_addr == 0
        or a_exc_out_addr == 0
        or i_exc_out_addr == 0
        or a_inh_out_addr == 0
        or i_inh_out_addr == 0
        or spikes_out_addr == 0
    ):
        return 1
    var final_addresses = [
        v_final_addr,
        a_exc_final_addr,
        i_exc_final_addr,
        a_inh_final_addr,
        i_inh_final_addr,
        spike_count_addr,
    ]
    for left in range(6):
        for right in range(left + 1, 6):
            if _ranges_overlap(final_addresses[left], 1, final_addresses[right], 1):
                return 1
    if steps > 0:
        var buffer_addresses = [
            exc_current_addr,
            inh_current_addr,
            v_out_addr,
            a_exc_out_addr,
            i_exc_out_addr,
            a_inh_out_addr,
            i_inh_out_addr,
            spikes_out_addr,
        ]
        for left in range(8):
            for right in range(left + 1, 8):
                if _ranges_overlap(
                    buffer_addresses[left], steps, buffer_addresses[right], steps
                ):
                    return 1
        for left in range(8):
            for right in range(6):
                if _ranges_overlap(buffer_addresses[left], steps, final_addresses[right], 1):
                    return 1
    if not _valid_configuration(
        v_init, a_exc_init, i_exc_init, a_inh_init, i_inh_init,
        v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt,
    ):
        return 2

    var v_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_final_addr)
    var a_exc_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=a_exc_final_addr
    )
    var i_exc_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_exc_final_addr
    )
    var a_inh_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=a_inh_final_addr
    )
    var i_inh_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_inh_final_addr
    )
    var spike_count = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=spike_count_addr
    )
    if steps == 0:
        if write_output:
            v_final[0], a_exc_final[0], i_exc_final[0] = v_init, a_exc_init, i_exc_init
            a_inh_final[0], i_inh_final[0], spike_count[0] = a_inh_init, i_inh_init, 0.0
        return 0

    var exc_current = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=exc_current_addr
    )
    var inh_current = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=inh_current_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_out_addr)
    var a_exc_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=a_exc_out_addr
    )
    var i_exc_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_exc_out_addr
    )
    var a_inh_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=a_inh_out_addr
    )
    var i_inh_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=i_inh_out_addr
    )
    var spikes_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )
    for index in range(steps):
        if not isfinite(exc_current[index]) or not isfinite(inh_current[index]):
            return 3

    var v = v_init
    var a_exc = a_exc_init
    var i_exc = i_exc_init
    var a_inh = a_inh_init
    var i_inh = i_inh_init
    var count = 0
    for index in range(steps):
        var a_exc_next = _rise_next(a_exc, exc_current[index], tau_exc, dt)
        var i_exc_next = _filter_next(a_exc, i_exc, exc_current[index], tau_exc, dt)
        var a_inh_next = _rise_next(a_inh, inh_current[index], tau_inh, dt)
        var i_inh_next = _filter_next(a_inh, i_inh, inh_current[index], tau_inh, dt)
        if (
            not isfinite(a_exc_next)
            or not isfinite(i_exc_next)
            or not isfinite(a_inh_next)
            or not isfinite(i_inh_next)
        ):
            return 4
        var exc_steady = tau_exc * exc_current[index]
        var inh_steady = tau_inh * inh_current[index]
        var v_steady = v_rest + exc_steady - inh_steady
        var decay_v = exp(-dt / tau_v)
        var v_next = v_steady + (v - v_steady) * decay_v + _drive_contribution(
            i_exc - exc_steady, a_exc - exc_steady, tau_exc, tau_v, dt
        ) - _drive_contribution(
            i_inh - inh_steady, a_inh - inh_steady, tau_inh, tau_v, dt
        )
        if not isfinite(v_next):
            return 4
        var spike = 0.0
        if v_next >= v_threshold:
            v = v_rest
            spike = 1.0
            count += 1
        else:
            v = v_next
        a_exc, i_exc = a_exc_next, i_exc_next
        a_inh, i_inh = a_inh_next, i_inh_next
        if write_output:
            v_out[index] = v
            a_exc_out[index] = a_exc
            i_exc_out[index] = i_exc
            a_inh_out[index] = a_inh
            i_inh_out[index] = i_inh
            spikes_out[index] = spike
    if write_output:
        v_final[0] = v
        a_exc_final[0] = a_exc
        i_exc_final[0] = i_exc
        a_inh_final[0] = a_inh
        i_inh_final[0] = i_inh
        spike_count[0] = Float64(count)
    return 0


@export
def alpha_simulate_c(
    n: Int32,
    v_init: Float64,
    a_exc_init: Float64,
    i_exc_init: Float64,
    a_inh_init: Float64,
    i_inh_init: Float64,
    v_rest: Float64,
    v_threshold: Float64,
    tau_v: Float64,
    tau_exc: Float64,
    tau_inh: Float64,
    dt: Float64,
    exc_current_addr: Int,
    inh_current_addr: Int,
    v_out_addr: Int,
    a_exc_out_addr: Int,
    i_exc_out_addr: Int,
    a_inh_out_addr: Int,
    i_inh_out_addr: Int,
    spikes_out_addr: Int,
    v_final_addr: Int,
    a_exc_final_addr: Int,
    i_exc_final_addr: Int,
    a_inh_final_addr: Int,
    i_inh_final_addr: Int,
    spike_count_addr: Int,
) -> Int32:
    var status = _run_alpha(
        n,
        v_init,
        a_exc_init,
        i_exc_init,
        a_inh_init,
        i_inh_init,
        v_rest,
        v_threshold,
        tau_v,
        tau_exc,
        tau_inh,
        dt,
        exc_current_addr,
        inh_current_addr,
        v_out_addr,
        a_exc_out_addr,
        i_exc_out_addr,
        a_inh_out_addr,
        i_inh_out_addr,
        spikes_out_addr,
        v_final_addr,
        a_exc_final_addr,
        i_exc_final_addr,
        a_inh_final_addr,
        i_inh_final_addr,
        spike_count_addr,
        False,
    )
    if status != 0:
        return status
    return _run_alpha(
        n,
        v_init,
        a_exc_init,
        i_exc_init,
        a_inh_init,
        i_inh_init,
        v_rest,
        v_threshold,
        tau_v,
        tau_exc,
        tau_inh,
        dt,
        exc_current_addr,
        inh_current_addr,
        v_out_addr,
        a_exc_out_addr,
        i_exc_out_addr,
        a_inh_out_addr,
        i_inh_out_addr,
        spikes_out_addr,
        v_final_addr,
        a_exc_final_addr,
        i_exc_final_addr,
        a_inh_final_addr,
        i_inh_final_addr,
        spike_count_addr,
        True,
    )
