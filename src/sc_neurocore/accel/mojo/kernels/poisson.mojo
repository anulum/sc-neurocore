# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo simulator for the seeded Poisson generator
#
# Build:
#   mojo build --emit shared-lib -o libpoisson.so poisson.mojo
#
# The caller supplies n_steps+1 Float64 slots: the binary event trace followed
# by the final LFSR16 state. A validation pass completes before any
# caller-visible output is written.

from std.math import exp, isfinite
from std.memory import UnsafePointer


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


def _run_poisson(
    rate_hz: Float64,
    dt_ms: Float64,
    rng_state0: Int,
    n_steps: Int,
    rate_override: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if (
        not isfinite(rate_hz)
        or rate_hz < 0.0
        or not isfinite(dt_ms)
        or dt_ms <= 0.0
        or not isfinite(rate_override)
        or rng_state0 <= 0
        or rng_state0 > 0xffff
    ):
        return -1

    var active_rate = rate_hz
    if rate_override >= 0.0:
        active_rate = rate_override
    if not isfinite(active_rate) or active_rate < 0.0:
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var rng_state = rng_state0
    var spikes: Int64 = 0
    for index in range(n_steps):
        var hazard = active_rate * dt_ms / 1000.0
        if not isfinite(hazard) or hazard < 0.0:
            return -1
        var probability = 1.0 - exp(-hazard)
        if not isfinite(probability) or probability < 0.0 or probability > 1.0:
            return -1
        var sample = _lfsr16_trial_sample(rng_state)
        var event = 0
        if sample < _probability_threshold(probability):
            event = 1
            spikes += 1
        rng_state = sample
        if write_output:
            output[index] = Float64(event)

    if write_output:
        output[n_steps] = Float64(rng_state)
    return spikes


@export
def poisson_simulate_c(
    rate_hz: Float64,
    dt_ms: Float64,
    rng_state: Int,
    n_steps: Int,
    rate_override: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0:
        return -1
    var validated = _run_poisson(
        rate_hz,
        dt_ms,
        rng_state,
        n_steps,
        rate_override,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_poisson(
        rate_hz,
        dt_ms,
        rng_state,
        n_steps,
        rate_override,
        output_addr,
        True,
    )
