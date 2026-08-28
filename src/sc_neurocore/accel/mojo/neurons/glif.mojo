# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Teeter 2018 GLIF5 source model
#
# Build: mojo build --emit shared-lib -o libglif.so glif.mojo
# The caller provides n+6 Float64 slots: voltage trace then complete final state.

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _convolution(decay_rate: Float64, forcing_rate: Float64, dt: Float64) -> Float64:
    var difference = decay_rate - forcing_rate
    var scale = max(1.0, max(abs(decay_rate), abs(forcing_rate)))
    if abs(difference) <= 1.0e-12 * scale:
        return dt * exp(-decay_rate * dt)
    return (exp(-forcing_rate * dt) - exp(-decay_rate * dt)) / difference


@export
def glif_simulate_c(
    v0: Float64,
    theta_spike0: Float64,
    i_asc1_0: Float64,
    i_asc2_0: Float64,
    theta_voltage0: Float64,
    refractory_remaining0: Float64,
    e_l: Float64,
    capacitance: Float64,
    resistance: Float64,
    theta_inf: Float64,
    b_spike: Float64,
    b_voltage: Float64,
    a_voltage: Float64,
    k_asc1: Float64,
    k_asc2: Float64,
    f_v: Float64,
    delta_v: Float64,
    delta_theta_spike: Float64,
    f_asc1: Float64,
    f_asc2: Float64,
    delta_i_asc1: Float64,
    delta_i_asc2: Float64,
    refractory_period: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(v0)
        or not isfinite(theta_spike0)
        or not isfinite(i_asc1_0)
        or not isfinite(i_asc2_0)
        or not isfinite(theta_voltage0)
        or not isfinite(refractory_remaining0)
        or not isfinite(e_l)
        or not isfinite(capacitance)
        or not isfinite(resistance)
        or not isfinite(theta_inf)
        or not isfinite(b_spike)
        or not isfinite(b_voltage)
        or not isfinite(a_voltage)
        or not isfinite(k_asc1)
        or not isfinite(k_asc2)
        or not isfinite(f_v)
        or not isfinite(delta_v)
        or not isfinite(delta_theta_spike)
        or not isfinite(f_asc1)
        or not isfinite(f_asc2)
        or not isfinite(delta_i_asc1)
        or not isfinite(delta_i_asc2)
        or not isfinite(refractory_period)
        or not isfinite(dt)
        or not isfinite(current)
        or capacitance <= 0.0
        or resistance <= 0.0
        or b_spike <= 0.0
        or b_voltage <= 0.0
        or k_asc1 <= 0.0
        or k_asc2 <= 0.0
        or dt <= 0.0
        or refractory_remaining0 < 0.0
        or refractory_period < 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var theta_spike = theta_spike0
    var i_asc1 = i_asc1_0
    var i_asc2 = i_asc2_0
    var theta_voltage = theta_voltage0
    var refractory_remaining = refractory_remaining0
    var membrane_rate = 1.0 / (resistance * capacitance)
    var membrane_decay = exp(-membrane_rate * dt)
    var spike_decay = exp(-b_spike * dt)
    var voltage_decay = exp(-b_voltage * dt)
    var asc1_decay = exp(-k_asc1 * dt)
    var asc2_decay = exp(-k_asc2 * dt)
    var voltage_convolution = _convolution(b_voltage, membrane_rate, dt)
    var events: Int64 = 0
    for index in range(n_steps):
        if refractory_remaining > 0.0:
            refractory_remaining = max(0.0, refractory_remaining - dt)
            trace[index] = v
            continue
        var total_current = current + i_asc1 + i_asc2
        var equilibrium_offset = resistance * total_current
        var voltage_offset = v - e_l
        var next_offset = equilibrium_offset + (voltage_offset - equilibrium_offset) * membrane_decay
        v = e_l + next_offset
        theta_spike *= spike_decay
        i_asc1 *= asc1_decay
        i_asc2 *= asc2_decay
        var threshold_forcing = equilibrium_offset * (1.0 - voltage_decay) / b_voltage
        threshold_forcing += (voltage_offset - equilibrium_offset) * voltage_convolution
        theta_voltage = theta_voltage * voltage_decay + a_voltage * threshold_forcing
        if (
            not isfinite(v)
            or not isfinite(theta_spike)
            or not isfinite(i_asc1)
            or not isfinite(i_asc2)
            or not isfinite(theta_voltage)
        ):
            return -1
        if v > theta_inf + theta_spike + theta_voltage:
            v = e_l + f_v * (v - e_l) - delta_v
            theta_spike += delta_theta_spike
            i_asc1 = f_asc1 * i_asc1 + delta_i_asc1
            i_asc2 = f_asc2 * i_asc2 + delta_i_asc2
            refractory_remaining = refractory_period
            events += 1
        if (
            not isfinite(v)
            or not isfinite(theta_spike)
            or not isfinite(i_asc1)
            or not isfinite(i_asc2)
            or not isfinite(theta_voltage)
        ):
            return -1
        trace[index] = v
    trace[n_steps] = v
    trace[n_steps + 1] = theta_spike
    trace[n_steps + 2] = i_asc1
    trace[n_steps + 3] = i_asc2
    trace[n_steps + 4] = theta_voltage
    trace[n_steps + 5] = refractory_remaining
    return events
