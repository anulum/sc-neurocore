# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo simulator for the seeded EscapeRate cell
#
# Build:
#   mojo build --emit shared-lib -o libescape_rate.so escape_rate.mojo
#
# The caller supplies 2*n_steps+2 Float64 slots: voltage trace, event trace,
# final voltage, and final LFSR16 state. A validation pass completes before any
# caller-visible output is written.

from std.math import exp, isfinite
from std.memory import UnsafePointer


def escape_rate_valid(
    v: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(v)
        and isfinite(v_rest)
        and isfinite(v_reset)
        and isfinite(v_threshold)
        and isfinite(tau_m)
        and tau_m > 0.0
        and isfinite(rho_0)
        and rho_0 > 0.0
        and isfinite(delta_u)
        and delta_u > 0.0
        and isfinite(resistance)
        and resistance > 0.0
        and isfinite(dt)
        and dt > 0.0
    )


def _safe_exp(x: Float64) -> Float64:
    if x > 700.0:
        return exp(700.0)
    if x < -700.0:
        return exp(-700.0)
    return exp(x)


@always_inline
def _lfsr16_advance(state: Int) -> Int:
    var feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return ((state >> 1) | (feedback << 15)) & 0xffff


@always_inline
def _lfsr16_trial_sample(state: Int) -> Int:
    var sample = state
    for _ in range(8):
        sample = _lfsr16_advance(sample)
    return sample


@always_inline
def _probability_threshold(probability: Float64) -> Int:
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return 65536
    return Int(probability * 65535.0) + 1


def _run_escape_rate(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
    rng_state0: Int,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if (
        not escape_rate_valid(
            v0,
            v_rest,
            v_reset,
            v_threshold,
            tau_m,
            rho_0,
            delta_u,
            resistance,
            dt,
        )
        or not isfinite(current)
        or rng_state0 <= 0
        or rng_state0 > 0xffff
    ):
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var v = v0
    var rng_state = rng_state0
    var spikes: Int64 = 0
    for index in range(n_steps):
        var v_inf = v_rest + resistance * current
        var decay = exp(-dt / tau_m)
        var next_v = v_inf + (v - v_inf) * decay
        if not isfinite(v_inf) or not isfinite(decay) or not isfinite(next_v):
            return -1
        var rate = rho_0 * _safe_exp((next_v - v_threshold) / delta_u)
        var hazard = rate * dt
        if not isfinite(hazard) or hazard < 0.0:
            return -1
        var probability = 1.0 - exp(-hazard)
        if not isfinite(probability) or probability < 0.0 or probability > 1.0:
            return -1
        var sample = _lfsr16_trial_sample(rng_state)
        var event = 0
        if sample < _probability_threshold(probability):
            event = 1
            next_v = v_reset
            spikes += 1
        rng_state = sample
        v = next_v
        if write_output:
            output[index] = v
            output[n_steps + index] = Float64(event)

    if write_output:
        output[2 * n_steps] = v
        output[2 * n_steps + 1] = Float64(rng_state)
    return spikes


@export
def escape_rate_simulate_c(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau_m: Float64,
    rho_0: Float64,
    delta_u: Float64,
    resistance: Float64,
    dt: Float64,
    rng_state: Int,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0:
        return -1
    var validated = _run_escape_rate(
        v0,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        rho_0,
        delta_u,
        resistance,
        dt,
        rng_state,
        n_steps,
        current,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_escape_rate(
        v0,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        rho_0,
        delta_u,
        resistance,
        dt,
        rng_state,
        n_steps,
        current,
        output_addr,
        True,
    )
