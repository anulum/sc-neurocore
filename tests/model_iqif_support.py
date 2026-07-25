# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared IQIF trace fixture

"""Shared trace construction for IQIF model contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron


def _trace(neuron: IntegerQIFNeuron, steps: int, current: int) -> tuple[list[int], list[int]]:
    values: list[int] = []
    spikes: list[int] = []
    for index in range(steps):
        if neuron.step(current):
            spikes.append(index)
        values.append(neuron.v)
    return values, spikes
