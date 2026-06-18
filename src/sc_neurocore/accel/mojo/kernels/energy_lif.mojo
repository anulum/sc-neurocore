# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration contract for energy-LIF exact flow

from std.math import exp

def _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


def energy_lif_valid(
    v: Float64,
    epsilon: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_e: Float64,
    alpha: Float64,
    epsilon_0: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and v >= -200.0
        and v <= 100.0
        and _finite(epsilon)
        and epsilon >= 0.0
        and _finite(v_rest)
        and _finite(v_reset)
        and v_reset >= -200.0
        and v_reset <= 100.0
        and _finite(v_threshold)
        and _finite(tau_m)
        and tau_m > 0.0
        and _finite(tau_e)
        and tau_e > 0.0
        and _finite(alpha)
        and alpha >= 0.0
        and _finite(epsilon_0)
        and epsilon_0 >= 0.0
        and _finite(resistance)
        and resistance > 0.0
        and _finite(dt)
        and dt > 0.0
        and epsilon <= epsilon_0
        and dt <= tau_m
        and dt <= tau_e
        and v_threshold > v_rest
        and v_threshold > v_reset
    )


def energy_lif_next_epsilon(
    v: Float64,
    epsilon: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_e: Float64,
    alpha: Float64,
    epsilon_0: Float64,
    resistance: Float64,
    dt: Float64,
    current: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not energy_lif_valid(v, epsilon, v_rest, v_reset, v_threshold, tau_m, tau_e, alpha, epsilon_0, resistance, dt):
        return 0.0 / 0.0
    return epsilon_0 + (epsilon - epsilon_0) * exp(-dt / tau_e)


def energy_lif_next_v(
    v: Float64,
    epsilon: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_e: Float64,
    alpha: Float64,
    epsilon_0: Float64,
    resistance: Float64,
    dt: Float64,
    current: Float64,
) -> Float64:
    if not _finite(current):
        return 0.0 / 0.0
    if not energy_lif_valid(v, epsilon, v_rest, v_reset, v_threshold, tau_m, tau_e, alpha, epsilon_0, resistance, dt):
        return 0.0 / 0.0
    var membrane_decay = exp(-dt / tau_m)
    var energy_delta = epsilon - epsilon_0
    var steady_energy_integral = epsilon_0 * tau_m * (1.0 - membrane_decay)
    var coupled_rate = (1.0 / tau_m) - (1.0 / tau_e)
    if coupled_rate < 1.0e-12 and coupled_rate > -1.0e-12:
        var transient_energy_integral = energy_delta * membrane_decay * dt
        return v_rest + (v - v_rest) * membrane_decay + (resistance * current / tau_m) * (steady_energy_integral + transient_energy_integral)
    var transient_energy_integral = energy_delta * membrane_decay * (exp(coupled_rate * dt) - 1.0) / coupled_rate
    return v_rest + (v - v_rest) * membrane_decay + (resistance * current / tau_m) * (steady_energy_integral + transient_energy_integral)


def energy_lif_step_spike(
    v: Float64,
    epsilon: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    tau_e: Float64,
    alpha: Float64,
    epsilon_0: Float64,
    resistance: Float64,
    dt: Float64,
    current: Float64,
) -> Int:
    var next_v = energy_lif_next_v(v, epsilon, v_rest, v_reset, v_threshold, tau_m, tau_e, alpha, epsilon_0, resistance, dt, current)
    var next_epsilon = energy_lif_next_epsilon(v, epsilon, v_rest, v_reset, v_threshold, tau_m, tau_e, alpha, epsilon_0, resistance, dt, current)
    if not (_finite(next_v) and _finite(next_epsilon)):
        return -1
    if not (next_v >= -200.0 and next_v <= 100.0 and next_epsilon >= 0.0 and next_epsilon <= epsilon_0):
        return -1
    if next_v >= v_threshold and next_epsilon > 0.1:
        var epsilon_after_spike = next_epsilon - alpha
        if epsilon_after_spike < 0.0:
            epsilon_after_spike = 0.0
        if not (_finite(epsilon_after_spike) and epsilon_after_spike <= epsilon_0):
            return -1
        return 1
    return 0
