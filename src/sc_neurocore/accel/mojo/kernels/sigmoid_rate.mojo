# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable exact-relaxation sigmoid-rate kernel and C ABI
#
# Build: mojo build --emit shared-lib -o libsigmoid_rate.so sigmoid_rate.mojo

from std.math import exp
from std.memory import UnsafePointer


@always_inline
def _sigmoid_rate_finite(value: Float64) -> Bool:
    var residual = value - value
    return value == value and residual == 0.0


@always_inline
def sigmoid_rate_valid(
    r: Float64, tau: Float64, beta: Float64, theta: Float64, dt: Float64
) -> Bool:
    return (
        _sigmoid_rate_finite(r)
        and _sigmoid_rate_finite(tau)
        and _sigmoid_rate_finite(beta)
        and _sigmoid_rate_finite(theta)
        and _sigmoid_rate_finite(dt)
        and r >= 0.0
        and r <= 1.0
        and tau > 0.0
        and dt > 0.0
    )


@always_inline
def _sigmoid_rate_transfer(
    beta: Float64, current: Float64, theta: Float64
) -> Float64:
    var argument = beta * (current - theta)
    if argument != argument:
        return -1.0
    if argument > 1.7976931348623157e308:
        return 1.0
    if argument < -1.7976931348623157e308:
        return 0.0
    if not _sigmoid_rate_finite(argument):
        return -1.0
    if argument >= 0.0:
        return 1.0 / (1.0 + exp(-argument))
    var exp_argument = exp(argument)
    return exp_argument / (1.0 + exp_argument)


@always_inline
def sigmoid_rate_step(
    r: Float64,
    current: Float64,
    tau: Float64,
    beta: Float64,
    theta: Float64,
    dt: Float64,
) -> Float64:
    if not _sigmoid_rate_finite(current):
        return -1.0
    if not sigmoid_rate_valid(r, tau, beta, theta, dt):
        return -1.0
    var target = _sigmoid_rate_transfer(beta, current, theta)
    if target < 0.0:
        return -1.0
    var decay = exp(-dt / tau)
    var next_r = decay * r + (1.0 - decay) * target
    if not _sigmoid_rate_finite(next_r) or next_r < 0.0 or next_r > 1.0:
        return -1.0
    return next_r


def _run_sigmoid_rate(
    r0: Float64,
    tau: Float64,
    beta: Float64,
    theta: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if n_steps < 0 or output_addr == 0:
        return -1
    if not sigmoid_rate_valid(r0, tau, beta, theta, dt):
        return -1
    if not _sigmoid_rate_finite(current):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var rate = r0
    for index in range(n_steps):
        rate = sigmoid_rate_step(rate, current, tau, beta, theta, dt)
        if rate < 0.0:
            return -1
        if write_output:
            output[index] = rate
    if write_output:
        output[n_steps] = rate
    return 0


@export
def sigmoid_rate_simulate_c(
    r: Float64,
    tau: Float64,
    beta: Float64,
    theta: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    var validated = _run_sigmoid_rate(
        r, tau, beta, theta, dt, n_steps, current, output_addr, False
    )
    if validated != 0:
        return -1
    return _run_sigmoid_rate(
        r, tau, beta, theta, dt, n_steps, current, output_addr, True
    )
