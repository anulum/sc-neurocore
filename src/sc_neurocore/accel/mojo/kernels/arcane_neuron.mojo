# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar helpers for arcane_neuron

from std.math import exp


fn _arcane_finite(x: Float64) -> Bool:
    var residual = x - x
    return x == x and residual == 0.0


fn arcane_valid_scalar_state(
    v_fast: Float64,
    v_work: Float64,
    v_deep: Float64,
    tau_fast: Float64,
    tau_work: Float64,
    tau_deep: Float64,
    theta: Float64,
    dt: Float64,
) -> Bool:
    return (
        _arcane_finite(v_fast)
        and _arcane_finite(v_work)
        and _arcane_finite(v_deep)
        and _arcane_finite(tau_fast)
        and _arcane_finite(tau_work)
        and _arcane_finite(tau_deep)
        and _arcane_finite(theta)
        and _arcane_finite(dt)
        and tau_fast > 0.0
        and tau_work > 0.0
        and tau_deep > 0.0
        and theta > 0.0
        and dt > 0.0
    )


fn arcane_exact_relaxation(
    state: Float64, steady_state: Float64, dt: Float64, tau: Float64
) -> Float64:
    if (
        not _arcane_finite(state)
        or not _arcane_finite(steady_state)
        or not _arcane_finite(dt)
        or not _arcane_finite(tau)
        or dt <= 0.0
        or tau <= 0.0
    ):
        return -1.0
    var decay = exp(-dt / tau)
    var candidate = decay * state + (1.0 - decay) * steady_state
    if not _arcane_finite(candidate):
        return -1.0
    return candidate


fn arcane_stable_sigmoid(x: Float64) -> Float64:
    if x != x:
        return -1.0
    if not _arcane_finite(x):
        if x > 0.0:
            return 1.0
        return 0.0
    if x >= 0.0:
        var z = exp(-x)
        return 1.0 / (1.0 + z)
    var exp_z = exp(x)
    return exp_z / (1.0 + exp_z)


fn arcane_fast_candidate(
    v_fast: Float64,
    current: Float64,
    confidence: Float64,
    spike_rate: Float64,
    w_gate_0: Float64,
    w_gate_1: Float64,
    w_gate_2: Float64,
    w_gate_3: Float64,
    v_work: Float64,
    w_inh: Float64,
    tau_fast: Float64,
    dt: Float64,
) -> Float64:
    if (
        not _arcane_finite(current)
        or not _arcane_finite(confidence)
        or not _arcane_finite(spike_rate)
        or not _arcane_finite(w_inh)
    ):
        return -1.0
    var gate_input = (
        w_gate_0 * current
        + w_gate_1 * v_fast
        + w_gate_2 * v_work
        + w_gate_3 * confidence
    )
    var gate = arcane_stable_sigmoid(gate_input)
    if gate < 0.0:
        return -1.0
    var fast_drive = gate * current - w_inh * spike_rate
    return arcane_exact_relaxation(v_fast, fast_drive, dt, tau_fast)


fn arcane_effective_threshold(
    theta: Float64,
    gamma: Float64,
    v_deep: Float64,
    delta_conf: Float64,
    confidence: Float64,
) -> Float64:
    if (
        not _arcane_finite(theta)
        or not _arcane_finite(gamma)
        or not _arcane_finite(v_deep)
        or not _arcane_finite(delta_conf)
        or not _arcane_finite(confidence)
        or theta <= 0.0
    ):
        return -1.0
    var threshold = (
        theta * (1.0 + gamma * v_deep) * (1.0 - delta_conf * confidence)
    )
    if not _arcane_finite(threshold):
        return -1.0
    if threshold < 0.1:
        return 0.1
    return threshold
