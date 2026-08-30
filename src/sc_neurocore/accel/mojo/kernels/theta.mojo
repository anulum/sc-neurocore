# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Theta exact-flow sequential simulator
#
# Build:
#   mojo build --emit shared-lib -o libtheta.so theta.mojo
#
# The caller supplies n_steps+1 Float64 slots: the post-step phase trace and
# final phase. Rejected contracts leave the buffer untouched because a
# validation pass completes before emission.

from std.math import atan, cos, exp, floor, sqrt, tan
from std.memory import UnsafePointer

comptime PI = 3.14159265358979323846


def _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


def theta_valid(theta: Float64, dt: Float64) -> Bool:
    return _finite(theta) and _finite(dt) and dt > 0.0


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _event_packet_representable(current: Float64, dt: Float64) -> Bool:
    return current <= 0.0 or sqrt(current) * dt <= PI


def theta_step_spike(theta: Float64, current: Float64, dt: Float64) -> Int:
    if not _finite(current):
        return -1
    if not theta_valid(theta, dt):
        return -1
    if not _event_packet_representable(current, dt):
        return -1

    var y = tan(theta / 2.0)
    if current > 0.0:
        var root_i = sqrt(current)
        var phase = atan(y / root_i)
        var next_phase = phase + root_i * dt
        if _abs(cos(next_phase)) <= 1.0e-15:
            return 1
        if next_phase >= PI / 2.0:
            return 1
        var next_y = root_i * tan(next_phase)
        if not _finite(next_y):
            return -1
        return 0
    if current == 0.0:
        var denominator = 1.0 - y * dt
        if denominator <= 0.0:
            return 1
        return 0

    var root_i_neg = sqrt(-current)
    if _abs(y + root_i_neg) <= 1.0e-15:
        return 0
    var ratio = (y - root_i_neg) / (y + root_i_neg)
    var evolved = ratio * exp(2.0 * root_i_neg * dt)
    var crossing_denominator = 1.0 - evolved
    if not _finite(evolved) or not _finite(crossing_denominator):
        return -1
    if (ratio < 1.0 and evolved >= 1.0) or _abs(
        crossing_denominator
    ) <= 1.0e-15:
        return 1
    return 0


def _wrap_phase(theta: Float64) -> Float64:
    var two_pi = 2.0 * PI
    var wrapped = theta + PI
    wrapped = wrapped - floor(wrapped / two_pi) * two_pi
    return wrapped - PI


def theta_next_theta(theta: Float64, current: Float64, dt: Float64) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not theta_valid(theta, dt):
        return 0.0 / 0.0
    if not _event_packet_representable(current, dt):
        return 0.0 / 0.0

    var y = tan(theta / 2.0)
    if current > 0.0:
        var root_i = sqrt(current)
        var phase = atan(y / root_i)
        var next_phase = phase + root_i * dt
        if _abs(cos(next_phase)) <= 1.0e-15:
            return -PI
        return _wrap_phase(2.0 * atan(root_i * tan(next_phase)))
    if current == 0.0:
        var denominator = 1.0 - y * dt
        if _abs(denominator) <= 1.0e-15:
            return -PI
        return _wrap_phase(2.0 * atan(y / denominator))

    var root_i_neg = sqrt(-current)
    if _abs(y + root_i_neg) <= 1.0e-15:
        return theta
    var ratio = (y - root_i_neg) / (y + root_i_neg)
    var evolved = ratio * exp(2.0 * root_i_neg * dt)
    var denominator_neg = 1.0 - evolved
    if not _finite(evolved) or not _finite(denominator_neg):
        return 0.0 / 0.0
    if (
        (ratio < 1.0 and evolved >= 1.0) or _abs(denominator_neg) <= 1.0e-15
    ) and _abs(denominator_neg) <= 1.0e-15:
        return -PI
    return _wrap_phase(
        2.0 * atan(root_i_neg * (1.0 + evolved) / denominator_neg)
    )


def _run_theta(
    theta0: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    var theta = theta0
    if (
        not _finite(current)
        or not theta_valid(theta, dt)
        or not _event_packet_representable(current, dt)
    ):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var spikes: Int64 = 0
    for index in range(n_steps):
        var spike = theta_step_spike(theta, current, dt)
        if spike < 0:
            return -1
        var next_theta = theta_next_theta(theta, current, dt)
        if not _finite(next_theta):
            return -1
        theta = next_theta
        spikes += Int64(spike)
        if write_output:
            output[index] = theta
    if write_output:
        output[n_steps] = theta
    return spikes


@export
def theta_simulate_c(
    theta0: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not _finite(current):
        return -1
    var validated = _run_theta(theta0, dt, n_steps, current, output_addr, False)
    if validated < 0:
        return -1
    return _run_theta(theta0, dt, n_steps, current, output_addr, True)


def _run_theta_complete(
    theta0: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    phase_addr: Int,
    event_addr: Int,
    write_output: Bool,
) -> Int64:
    var theta = theta0
    if (
        not _finite(current)
        or not theta_valid(theta, dt)
        or not _event_packet_representable(current, dt)
    ):
        return -1
    var phase = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=phase_addr
    )
    var events = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=event_addr
    )
    var event_count: Int64 = 0
    for index in range(n_steps):
        var event = theta_step_spike(theta, current, dt)
        if event < 0:
            return -1
        var next_theta = theta_next_theta(theta, current, dt)
        if not _finite(next_theta):
            return -1
        theta = next_theta
        event_count += Int64(event)
        if write_output:
            phase[index] = theta
            events[index] = UInt8(event)
    if write_output:
        phase[n_steps] = theta
    return event_count


@export
def theta_simulate_complete_c(
    theta0: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    phase_addr: Int,
    event_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or phase_addr == 0
        or (n_steps > 0 and event_addr == 0)
        or not _finite(current)
    ):
        return -1
    var validated = _run_theta_complete(
        theta0, dt, n_steps, current, phase_addr, event_addr, False
    )
    if validated < 0:
        return -1
    return _run_theta_complete(
        theta0, dt, n_steps, current, phase_addr, event_addr, True
    )
