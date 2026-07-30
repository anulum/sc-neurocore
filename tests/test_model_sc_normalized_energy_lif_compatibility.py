# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Frozen compatibility checks for normalized energy LIF

"""Ensure the pre-Model-52 project recurrence remains intact."""

from __future__ import annotations

import hashlib
import struct

from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron,
)


def test_frozen_256_step_compatibility_trace() -> None:
    neuron = SCNormalizedEnergyLIFNeuron()
    payload = bytearray()
    events = 0
    for index in range(256):
        current = (30.0, 0.0, 50.0, 10.0)[index % 4]
        event = neuron.step(current)
        events += event
        payload.extend(struct.pack("<ddi", neuron.v, neuron.epsilon, event))
    assert events == 3
    assert (
        hashlib.sha256(payload).hexdigest()
        == "29a0793719677083d6502511382bee812c5ea1fcd0b8599cd6bb94c34187d12a"
    )


def test_retained_exact_flow_and_reset() -> None:
    neuron = SCNormalizedEnergyLIFNeuron(epsilon=0.5)
    expected = neuron._exact_candidate(10.0)
    assert neuron.step(10.0) == 0
    assert (neuron.v, neuron.epsilon) == expected
    neuron.reset()
    assert (neuron.v, neuron.epsilon) == (neuron.v_rest, neuron.epsilon_0)
