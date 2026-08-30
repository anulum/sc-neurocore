# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Quadratic IF sequential simulator
#
# Build:
#   mojo build --emit shared-lib -o libquadratic_if.so quadratic_if.mojo
#
# The caller supplies n_steps+1 Float64 slots: the post-step voltage trace and
# final voltage. Rejected contracts leave the buffer untouched because a
# validation pass completes before emission.

from std.math import atan, exp, sqrt, tan
from std.memory import UnsafePointer


def _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def quadratic_if_valid(
    v: Float64, v_reset: Float64, v_peak: Float64, dt: Float64
) -> Bool:
    return (
        _finite(v)
        and _finite(v_reset)
        and _finite(v_peak)
        and _finite(dt)
        and v < v_peak
        and v_reset < v_peak
        and dt > 0.0
    )


def quadratic_if_step_spike(
    v: Float64, current: Float64, v_reset: Float64, v_peak: Float64, dt: Float64
) -> Int:
    if not _finite(current):
        return -1
    if not quadratic_if_valid(v, v_reset, v_peak, dt):
        return -1

    if current > 0.0:
        var root_i = sqrt(current)
        var phase = atan(v / root_i)
        var peak_phase = atan(v_peak / root_i)
        var next_phase = phase + root_i * dt
        if next_phase >= peak_phase or next_phase >= 1.5707963267948966:
            return 1
        var next_pos = root_i * tan(next_phase)
        if not _finite(next_pos):
            return -1
        if next_pos >= v_peak:
            return 1
        return 0
    if current == 0.0:
        var denominator = 1.0 - v * dt
        if denominator <= 0.0:
            return 1
        var next_zero = v / denominator
        if not _finite(next_zero):
            return -1
        if next_zero >= v_peak:
            return 1
        return 0

    var root_neg_i = sqrt(-current)
    if _abs(v + root_neg_i) <= 0.000000000000001:
        return 0
    var numerator_ratio = (v - root_neg_i) / (v + root_neg_i)
    var evolved_ratio = numerator_ratio * exp(2.0 * root_neg_i * dt)
    var denom = 1.0 - evolved_ratio
    if (numerator_ratio < 1.0 and evolved_ratio >= 1.0) or _abs(denom) <= 0.000000000000001:
        return 1
    var next_neg = root_neg_i * (1.0 + evolved_ratio) / denom
    if not _finite(next_neg):
        return -1
    if next_neg >= v_peak:
        return 1
    return 0


def quadratic_if_next_v(
    v: Float64, current: Float64, v_reset: Float64, v_peak: Float64, dt: Float64
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not quadratic_if_valid(v, v_reset, v_peak, dt):
        return 0.0 / 0.0

    if current > 0.0:
        var root_i = sqrt(current)
        var phase = atan(v / root_i)
        var peak_phase = atan(v_peak / root_i)
        var next_phase = phase + root_i * dt
        if next_phase >= peak_phase or next_phase >= 1.5707963267948966:
            return v_reset
        return root_i * tan(next_phase)
    if current == 0.0:
        var denominator = 1.0 - v * dt
        if denominator <= 0.0:
            return v_reset
        var next_zero = v / denominator
        if next_zero >= v_peak:
            return v_reset
        return next_zero

    var root_neg_i = sqrt(-current)
    if _abs(v + root_neg_i) <= 0.000000000000001:
        return v
    var numerator_ratio = (v - root_neg_i) / (v + root_neg_i)
    var evolved_ratio = numerator_ratio * exp(2.0 * root_neg_i * dt)
    var denom = 1.0 - evolved_ratio
    if (numerator_ratio < 1.0 and evolved_ratio >= 1.0) or _abs(denom) <= 0.000000000000001:
        return v_reset
    var next_neg = root_neg_i * (1.0 + evolved_ratio) / denom
    if next_neg >= v_peak:
        return v_reset
    return next_neg


def _run_quadratic_if(
    v0: Float64,
    v_reset: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    if not _finite(current) or not quadratic_if_valid(v, v_reset, v_peak, dt):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var spikes: Int64 = 0
    for index in range(n_steps):
        var spike = quadratic_if_step_spike(v, current, v_reset, v_peak, dt)
        if spike < 0:
            return -1
        var next_v = quadratic_if_next_v(v, current, v_reset, v_peak, dt)
        if not _finite(next_v):
            return -1
        v = next_v
        spikes += Int64(spike)
        if write_output:
            output[index] = v
    if write_output:
        output[n_steps] = v
    return spikes


@export
def quadratic_if_simulate_c(
    v0: Float64,
    v_reset: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not _finite(current):
        return -1
    var validated = _run_quadratic_if(
        v0, v_reset, v_peak, dt, n_steps, current, output_addr, False
    )
    if validated < 0:
        return -1
    return _run_quadratic_if(
        v0, v_reset, v_peak, dt, n_steps, current, output_addr, True
    )


def _run_quadratic_if_complete(
    v0: Float64,
    v_reset: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    voltage_addr: Int,
    event_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    if not _finite(current) or not quadratic_if_valid(v, v_reset, v_peak, dt):
        return -1
    var voltage = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=voltage_addr
    )
    var events = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=event_addr)
    var event_count: Int64 = 0
    for index in range(n_steps):
        var event = quadratic_if_step_spike(v, current, v_reset, v_peak, dt)
        if event < 0:
            return -1
        var next_v = quadratic_if_next_v(v, current, v_reset, v_peak, dt)
        if not _finite(next_v):
            return -1
        v = next_v
        event_count += Int64(event)
        if write_output:
            voltage[index] = v
            events[index] = UInt8(event)
    if write_output:
        voltage[n_steps] = v
    return event_count


@export
def quadratic_if_simulate_complete_c(
    v0: Float64,
    v_reset: Float64,
    v_peak: Float64,
    dt: Float64,
    source_profile: Int,
    n_steps: Int,
    current: Float64,
    voltage_addr: Int,
    event_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or voltage_addr == 0
        or (n_steps > 0 and event_addr == 0)
        or (source_profile != 0 and source_profile != 1)
        or not _finite(current)
    ):
        return -1
    var validated = _run_quadratic_if_complete(
        v0, v_reset, v_peak, dt, n_steps, current,
        voltage_addr, event_addr, False,
    )
    if validated < 0:
        return -1
    return _run_quadratic_if_complete(
        v0, v_reset, v_peak, dt, n_steps, current,
        voltage_addr, event_addr, True,
    )
