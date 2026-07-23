# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator co-simulation references

"""Independent Perfect Integrator spike-count and sawtooth reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from tests.cosim_reference_statistics import _summarise


def _perfect_integrator_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored perfect-integrator spike count for comparison."""
    neuron = PerfectIntegratorNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _perfect_integrator_sawtooth_features(
    *,
    current: float,
    dt: float,
    steps: int,
    c_m: float = 1.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> dict[str, float]:
    """Return exact post-reset features for constant-current perfect integration."""
    values: list[float] = []
    spikes: list[int] = []
    voltage = v_reset
    increment = current * dt / c_m
    for _ in range(steps):
        voltage += increment
        if voltage >= v_threshold:
            spikes.append(1)
            voltage = v_reset
        else:
            spikes.append(0)
        values.append(voltage)

    return _summarise({"v": values}, spikes)
