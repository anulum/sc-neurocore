# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for energy_lif

fn valid(v: Float64, epsilon: Float64, v_rest: Float64, v_reset: Float64, v_threshold: Float64, tau_m: Float64, tau_e: Float64, alpha: Float64, epsilon_0: Float64, resistance: Float64, dt: Float64) -> Bool:
    return v.is_finite() and epsilon.is_finite() and epsilon >= 0.0 and v_rest.is_finite() and v_reset.is_finite() and v_threshold.is_finite() and tau_m.is_finite() and tau_m > 0.0 and tau_e.is_finite() and tau_e > 0.0 and alpha.is_finite() and alpha >= 0.0 and epsilon_0.is_finite() and epsilon_0 >= 0.0 and resistance.is_finite() and resistance > 0.0 and dt.is_finite() and dt > 0.0 and epsilon <= epsilon_0 and dt <= tau_m and dt <= tau_e and v_threshold > v_rest and v_threshold > v_reset

fn step(v: Float64, epsilon: Float64, v_rest: Float64, v_reset: Float64, v_threshold: Float64, tau_m: Float64, tau_e: Float64, alpha: Float64, epsilon_0: Float64, resistance: Float64, dt: Float64, current: Float64) -> Int:
    if not valid(v, epsilon, v_rest, v_reset, v_threshold, tau_m, tau_e, alpha, epsilon_0, resistance, dt) or not current.is_finite():
        return 0
    var effective_r = resistance * epsilon
    var next_v = v + (-(v - v_rest) + effective_r * current) / tau_m * dt
    var next_epsilon = epsilon + (epsilon_0 - epsilon) / tau_e * dt
    if next_v >= v_threshold and next_epsilon > 0.1:
        return 1
    return 0

fn reset() -> Int:
    return 0
