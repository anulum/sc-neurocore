# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta reference-trace contracts

"""Theta reference-trace parity contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.theta import ThetaNeuron
from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _theta_constant_current_features


def test_theta_trace_features_match_independent_phase_solution() -> None:
    """Committed theta features must match the tangent half-angle phase solution."""
    spec = load_reference_trace_spec("theta_constant_current_phase_analytic")

    expected = _theta_constant_current_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "theta"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_theta_source_receipt_matches_digests_and_independent_closed_form() -> None:
    """Bind equation (2.5) to input, phase, event, and circle-flow evidence."""
    path = (
        Path(__file__).parents[1]
        / "src/sc_neurocore/neurons/reference_receipts/theta_ermentrout_kopell_1986.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    numerical = receipt["numerical_specialization"]
    expected = receipt["expected"]
    current = float(protocol["current"])
    steps = int(protocol["steps"])
    dt = float(numerical["dt"])
    initial_theta = float(numerical["initial_theta"])
    inputs = np.full(steps, current, dtype="<f8")
    neuron = ThetaNeuron(theta=initial_theta, dt=dt)
    phase, events = neuron.simulate_complete(steps, current, "python")

    root = math.sqrt(current)
    transformed_initial = math.atan(math.tan(initial_theta / 2.0) / root)
    indices = np.arange(1, steps + 1, dtype=np.float64)
    unwrapped = transformed_initial + root * dt * indices
    oracle_phase = (2.0 * np.arctan(root * np.tan(unwrapped)) + math.pi) % (2.0 * math.pi) - math.pi
    previous = transformed_initial + root * dt * np.arange(steps, dtype=np.float64)
    oracle_events = (
        np.floor((unwrapped + math.pi / 2.0) / math.pi)
        > np.floor((previous + math.pi / 2.0) / math.pi)
    ).astype(np.uint8)
    circular_error = np.abs((phase - oracle_phase + math.pi) % (2.0 * math.pi) - math.pi)

    assert receipt["reference"]["pdf_sha256"] == (
        "4b733c80d40e4d8cee306366b05bf9475ec6dff71957d4d53b96acadaf092d51"
    )
    assert (
        hashlib.sha256(inputs.tobytes()).hexdigest()
        == (protocol["little_endian_float64_input_sha256"])
    )
    assert (
        hashlib.sha256(phase.astype("<f8").tobytes()).hexdigest()
        == (expected["little_endian_float64_phase_sha256"])
    )
    assert hashlib.sha256(events.tobytes()).hexdigest() == expected["uint8_event_sha256"]
    np.testing.assert_array_equal(events, oracle_events)
    assert float(circular_error.max()) <= expected["independent_closed_form_max_circular_error"]
    assert np.flatnonzero(events).tolist() == expected["event_indices_zero_based"]
    assert int(events.sum()) == expected["event_count"]
    assert neuron.theta == expected["final_theta"]
