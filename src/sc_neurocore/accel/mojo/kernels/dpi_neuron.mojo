# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo simulator for the published DPI neuron circuit
#
# Build:
#   mojo build --emit shared-lib -o libdpi_neuron.so dpi_neuron.mojo
#
# The caller supplies n_steps+3 Float64 slots: the post-step membrane-current
# trace followed by final i_mem, i_ahp, and refractory_time. Validation completes
# before any output write.

from std.math import exp, isfinite, log
from std.memory import UnsafePointer


@always_inline
def _positive(value: Float64) -> Bool:
    return isfinite(value) and value > 0.0


@always_inline
def _sigmoid(value: Float64) -> Float64:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    var exponential = exp(value)
    return exponential / (1.0 + exponential)


def dpi_valid(
    i_mem: Float64,
    i_ahp: Float64,
    refractory_time: Float64,
    i_threshold: Float64,
    i_reset: Float64,
    i_rest: Float64,
    i_tau: Float64,
    i_g: Float64,
    i_tau_ahp: Float64,
    i_ga: Float64,
    i_spike: Float64,
    i_0: Float64,
    kappa: Float64,
    alpha: Float64,
    tau: Float64,
    tau_ahp: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Bool:
    return (
        _positive(i_mem)
        and isfinite(i_ahp)
        and i_ahp >= 0.0
        and isfinite(refractory_time)
        and refractory_time >= 0.0
        and _positive(i_threshold)
        and _positive(i_reset)
        and i_reset < i_threshold
        and isfinite(i_rest)
        and i_rest >= 0.0
        and _positive(i_tau)
        and _positive(i_g)
        and _positive(i_tau_ahp)
        and _positive(i_ga)
        and _positive(i_spike)
        and _positive(i_0)
        and _positive(kappa)
        and _positive(alpha)
        and _positive(tau)
        and _positive(tau_ahp)
        and _positive(refractory_period)
        and _positive(dt)
        and refractory_period >= dt
    )


def _run_dpi(
    i_mem0: Float64,
    i_ahp0: Float64,
    refractory_time0: Float64,
    i_threshold: Float64,
    i_reset: Float64,
    i_rest: Float64,
    i_tau: Float64,
    i_g: Float64,
    i_tau_ahp: Float64,
    i_ga: Float64,
    i_spike: Float64,
    i_0: Float64,
    kappa: Float64,
    alpha: Float64,
    tau: Float64,
    tau_ahp: Float64,
    refractory_period: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    var total_input = i_rest + current
    if (
        not dpi_valid(
            i_mem0,
            i_ahp0,
            refractory_time0,
            i_threshold,
            i_reset,
            i_rest,
            i_tau,
            i_g,
            i_tau_ahp,
            i_ga,
            i_spike,
            i_0,
            kappa,
            alpha,
            tau,
            tau_ahp,
            refractory_period,
            dt,
        )
        or not isfinite(current)
        or not isfinite(total_input)
        or total_input < 0.0
    ):
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var i_mem = i_mem0
    var i_ahp = i_ahp0
    var refractory_time = refractory_time0
    var spikes: Int64 = 0
    for index in range(n_steps):
        var spike_active = refractory_time > 0.0
        var spike_current = 0.0
        if spike_active:
            spike_current = i_spike
        var d_i_ahp = (
            i_ahp
            / (tau_ahp * i_tau_ahp)
            * (spike_current / (1.0 + i_ahp / i_ga) - i_tau_ahp)
        )
        var next_i_ahp = i_ahp + dt * d_i_ahp
        var next_i_mem = i_reset
        var next_refractory = 0.0
        if spike_active:
            next_refractory = refractory_time - dt
            if next_refractory < 0.0:
                next_refractory = 0.0
        else:
            var log_current = (log(i_0) + kappa * log(i_mem)) / (kappa + 1.0)
            var gate = _sigmoid(alpha * (i_mem - i_threshold))
            var i_fb = exp(log_current) * gate
            var d_i_mem = (
                i_mem
                / (tau * i_tau)
                * (total_input / (1.0 + i_mem / i_g) - i_tau + i_fb - i_ahp)
            )
            next_i_mem = i_mem + dt * d_i_mem
            if not isfinite(next_i_mem) or next_i_mem <= 0.0:
                return -1
            if next_i_mem >= i_threshold:
                next_i_mem = i_reset
                next_refractory = refractory_period
                spikes += 1
        if not (
            isfinite(next_i_mem)
            and isfinite(next_i_ahp)
            and isfinite(next_refractory)
            and next_i_mem > 0.0
            and next_i_ahp >= 0.0
            and next_refractory >= 0.0
        ):
            return -1
        i_mem = next_i_mem
        i_ahp = next_i_ahp
        refractory_time = next_refractory
        if write_output:
            output[index] = i_mem
    if write_output:
        output[n_steps] = i_mem
        output[n_steps + 1] = i_ahp
        output[n_steps + 2] = refractory_time
    return spikes


@export
def dpi_neuron_simulate_c(
    i_mem0: Float64,
    i_ahp0: Float64,
    refractory_time0: Float64,
    i_threshold: Float64,
    i_reset: Float64,
    i_rest: Float64,
    i_tau: Float64,
    i_g: Float64,
    i_tau_ahp: Float64,
    i_ga: Float64,
    i_spike: Float64,
    i_0: Float64,
    kappa: Float64,
    alpha: Float64,
    tau: Float64,
    tau_ahp: Float64,
    refractory_period: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not isfinite(current):
        return -1
    var validated = _run_dpi(
        i_mem0,
        i_ahp0,
        refractory_time0,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
        n_steps,
        current,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_dpi(
        i_mem0,
        i_ahp0,
        refractory_time0,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
        n_steps,
        current,
        output_addr,
        True,
    )
