# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Independent receipts for the source and retained SC McKean identities."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.neurons.reference_traces import reference_trace_spec_from_payload
from tests.cosim_support import _mckean_rk4_features


def test_source_mckean_receipt_matches_production() -> None:
    receipt = json.loads(
        (
            Path(__file__).parents[1]
            / "src/sc_neurocore/neurons/reference_receipts/mckean_tonnelier.json"
        ).read_text()
    )
    neuron = McKeanNeuron()
    digest = hashlib.sha256()
    events = 0
    pattern = tuple(receipt["drive"]["pattern"])
    for index in range(receipt["oracle"]["steps"]):
        event = neuron.step(pattern[index % len(pattern)])
        events += event
        digest.update(struct.pack("<ddB", neuron.v, neuron.w, event))
    assert events == receipt["oracle"]["events"]
    assert neuron.v == receipt["oracle"]["v_final"]
    assert neuron.w == receipt["oracle"]["w_final"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]


def test_sc_triangular_receipt_matches_independent_reference() -> None:
    payload = json.loads(
        (
            Path(__file__).parents[1]
            / "src/sc_neurocore/neurons/reference_trace_data/sc_triangular_mckean_project.json"
        ).read_text()
    )
    spec = reference_trace_spec_from_payload(payload)
    expected = _mckean_rk4_features(
        current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
    )
    assert spec.schema_name == "sc_triangular_mckean"
    assert spec.provenance.citation == "SC-NeuroCore project recurrence"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
