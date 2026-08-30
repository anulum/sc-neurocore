# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF reference-trace contracts

"""Quadratic IF reference-trace parity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.quadratic_if import (
    QuadraticIFNeuron,
    SCSymmetricQuadraticIFNeuron,
)
from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _quadratic_if_zero_current_features


def test_quadratic_if_trace_features_match_independent_analytic_solution() -> None:
    """Committed QIF features must match the analytic zero-current Riccati flow."""
    spec = load_reference_trace_spec("quadratic_if_zero_current_analytic")

    expected = _quadratic_if_zero_current_features(
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "quadratic_if"
    assert spec.provenance.citation == "doi:10.1152/jn.2000.83.2.808"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_sc_symmetric_trace_retains_historical_analytic_protocol() -> None:
    """Keep the old -1/+1/.01 oracle under its count-neutral SC identity."""
    spec = load_reference_trace_spec("sc_symmetric_quadratic_if_zero_current_analytic")
    expected = _quadratic_if_zero_current_features(dt=0.01, steps=120)
    assert spec.schema_name == "sc_symmetric_quadratic_if"
    assert spec.provenance.citation == "SC-NeuroCore retained project recurrence"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_latham_source_receipt_matches_independent_digests() -> None:
    """Bind the normalized source profile to input, voltage, and event bytes."""
    path = (
        Path(__file__).parents[1]
        / "src/sc_neurocore/neurons/reference_receipts/quadratic_if_latham_2000.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    expected = receipt["expected"]
    current = float(protocol["eta"])
    steps = int(protocol["steps"])
    inputs = np.full(steps, current, dtype="<f8")
    neuron = QuadraticIFNeuron.latham_2000()
    voltage, events = neuron.simulate_complete(steps, current, "python")

    assert receipt["reference"]["pdf_sha256"] == (
        "be4c439a06b5609d499b2b3f8d631ed8b3adf233e66defd3957487300d5d42a0"
    )
    assert (
        hashlib.sha256(inputs.tobytes()).hexdigest()
        == protocol["little_endian_float64_input_sha256"]
    )
    assert (
        hashlib.sha256(voltage.astype("<f8").tobytes()).hexdigest()
        == expected["little_endian_float64_voltage_sha256"]
    )
    assert hashlib.sha256(events.tobytes()).hexdigest() == expected["uint8_event_sha256"]
    assert int(events.sum()) == expected["event_count"]
    assert neuron.v == expected["final_voltage"]


def test_sc_identity_is_not_the_latham_source_profile() -> None:
    """Preserve both public identities without silently changing legacy defaults."""
    source = QuadraticIFNeuron.latham_2000()
    retained = SCSymmetricQuadraticIFNeuron()
    assert (source.v_reset, source.v_peak, source.dt) == (-3.0, 31.0 / 3.0, 0.05)
    assert (retained.v_reset, retained.v_peak, retained.dt) == (-1.0, 1.0, 0.01)
