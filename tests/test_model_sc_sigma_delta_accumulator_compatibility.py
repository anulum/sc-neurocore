# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import hashlib
import struct

from sc_neurocore.neurons.models.sc_sigma_delta_accumulator import (
    SCSigmaDeltaAccumulatorNeuron,
)


def test_frozen_project_trace_receipt() -> None:
    drive = [0.0] * 32 + [0.3] * 96 + [-0.7, 1.1] * 64
    neuron = SCSigmaDeltaAccumulatorNeuron()
    digest = hashlib.sha256()
    events: list[int] = []
    for current in drive:
        event = neuron.step(current)
        events.append(event)
        digest.update(struct.pack("<di", neuron.sigma, event))
    assert events.count(1) == 54
    assert events.count(-1) == 0
    assert neuron.sigma == 0.40000000000000857
    assert digest.hexdigest() == "8cb57c49fe0bbd0c0a9d62bf888031883958daf91381e396a334dcdfe44f1e7a"


def test_signed_events_and_one_quantum_per_sample() -> None:
    neuron = SCSigmaDeltaAccumulatorNeuron()
    assert neuron.step(3.25) == 1
    assert neuron.sigma == 2.25
    assert neuron.step(-4.5) == -1
    assert neuron.sigma == -1.25
