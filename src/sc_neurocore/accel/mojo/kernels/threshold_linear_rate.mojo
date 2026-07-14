# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable threshold-linear rate kernel and C ABI
#
# Build: mojo build --emit shared-lib -o libthreshold_linear_rate.so threshold_linear_rate.mojo

from std.memory import UnsafePointer


@always_inline
def _threshold_linear_rate_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


@always_inline
def threshold_linear_rate_valid(r: Float64, theta: Float64, gain: Float64) -> Bool:
    return (
        _threshold_linear_rate_finite(r)
        and r >= 0.0
        and _threshold_linear_rate_finite(theta)
        and _threshold_linear_rate_finite(gain)
        and gain >= 0.0
    )


@always_inline
def threshold_linear_rate_step(
    r: Float64, theta: Float64, gain: Float64, current: Float64
) -> Float64:
    if not threshold_linear_rate_valid(r, theta, gain):
        return -1.0
    if not _threshold_linear_rate_finite(current):
        return -1.0
    var drive = current - theta
    if drive < 0.0:
        drive = 0.0
    var next_r = gain * drive
    if not _threshold_linear_rate_finite(next_r) or next_r < 0.0:
        return -1.0
    return next_r


def _run_threshold_linear_rate(
    r0: Float64,
    theta: Float64,
    gain: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if n_steps < 0 or output_addr == 0:
        return -1
    if not threshold_linear_rate_valid(r0, theta, gain):
        return -1
    if not _threshold_linear_rate_finite(current):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    var rate = r0
    for index in range(n_steps):
        rate = threshold_linear_rate_step(rate, theta, gain, current)
        if rate < 0.0:
            return -1
        if write_output:
            output[index] = rate
    if write_output:
        output[n_steps] = rate
    return 0


@export
def threshold_linear_rate_simulate_c(
    r: Float64,
    theta: Float64,
    gain: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    var validated = _run_threshold_linear_rate(
        r, theta, gain, n_steps, current, output_addr, False
    )
    if validated != 0:
        return -1
    return _run_threshold_linear_rate(
        r, theta, gain, n_steps, current, output_addr, True
    )
