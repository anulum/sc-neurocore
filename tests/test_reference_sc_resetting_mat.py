# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - SC resetting-MAT project trace receipt

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

RECEIPT = Path("src/sc_neurocore/neurons/reference_trace_data/sc_resetting_mat_project.json")


def test_sc_resetting_mat_trace_matches_presplit_binary_contract() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    currents = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    neuron = SCResettingMATNeuron()
    encoded = bytearray()
    events = 0
    for current in currents:
        event = neuron.step(current)
        events += event
        encoded.extend(struct.pack("<dddB", neuron.v, neuron.theta1, neuron.theta2, event))
    oracle = receipt["oracle"]
    assert hashlib.sha256(encoded).hexdigest() == oracle["trace_sha256"]
    assert events == oracle["event_count"]
    assert [neuron.v, neuron.theta1, neuron.theta2] == pytest.approx(
        oracle["final_state"], abs=1.0e-15
    )
