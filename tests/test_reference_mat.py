# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - source MAT* independent trace receipt

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from sc_neurocore.neurons.models.mat import MATNeuron

RECEIPT = Path("src/sc_neurocore/neurons/reference_trace_data/mat_2009_rs.json")


def test_source_mat_trace_matches_committed_independent_oracle() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    currents = [0.0] * 32 + [0.7] * 8192 + [0.2, 0.9] * 1024
    neuron = MATNeuron()
    encoded = bytearray()
    event_indices: list[int] = []
    for index, current in enumerate(currents):
        event = neuron.step(current)
        if event:
            event_indices.append(index)
        encoded.extend(
            struct.pack(
                "<ddddB",
                neuron.v,
                neuron.theta1,
                neuron.theta2,
                neuron.refractory_remaining,
                event,
            )
        )
    oracle = receipt["oracle"]
    assert hashlib.sha256(encoded).hexdigest() == oracle["trace_sha256"]
    assert event_indices == oracle["event_indices"]
    assert [neuron.v, neuron.theta1, neuron.theta2, neuron.refractory_remaining] == pytest.approx(
        oracle["final_state"], abs=1.0e-15
    )
