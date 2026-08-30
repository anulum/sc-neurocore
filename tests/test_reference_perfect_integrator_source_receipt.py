# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Naud-Gerstner perfect-integrator receipt oracle

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.perfect_integrator import (
    PerfectIntegratorNeuron,
    SCInclusivePerfectIntegratorNeuron,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "src/sc_neurocore/neurons/reference_receipts/perfect_integrator_naud_gerstner_2012.json"
)


def _sha256(array: npt.NDArray[np.generic]) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _independent_packet(
    receipt: dict[str, object], *, inclusive: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    numerical = receipt["numerical_specialization"]
    protocol = receipt["protocol"]
    assert isinstance(numerical, dict)
    assert isinstance(protocol, dict)
    voltage = np.empty(int(protocol["steps"]), dtype="<f8")
    events = np.zeros(int(protocol["steps"]), dtype="u1")
    v = float(numerical["v_initial"])
    increment = (
        float(protocol["current"]) * float(numerical["dt"]) / float(numerical["capacitance_C"])
    )
    for index in range(voltage.size):
        candidate = v + increment
        event = (
            candidate >= float(numerical["v_threshold"])
            if inclusive
            else candidate > float(numerical["v_threshold"])
        )
        events[index] = event
        v = float(numerical["v_reset"]) if event else candidate
        voltage[index] = v
    return voltage, events


def test_receipt_reproduces_from_independent_source_equations() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt["expected"]
    voltage, events = _independent_packet(receipt)
    indices = np.flatnonzero(events)
    assert int(events.sum()) == expected["event_count"]
    assert int(indices[0]) == expected["first_event_index_zero_based"]
    assert np.all(np.diff(indices) == expected["event_period_steps"])
    assert int(indices[-1]) == expected["last_event_index_zero_based"]
    assert float(voltage[-1]) == expected["final_voltage"]
    assert _sha256(voltage) == expected["little_endian_float64_voltage_sha256"]
    assert _sha256(events) == expected["uint8_event_sha256"]


def test_receipt_matches_production_source_complete_packet() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    expected = receipt["expected"]
    neuron = PerfectIntegratorNeuron.naud_gerstner_2012()
    voltage, events = neuron.simulate_complete(
        protocol["steps"], protocol["current"], backend="python"
    )
    assert (
        _sha256(voltage.astype("<f8", copy=False))
        == expected["little_endian_float64_voltage_sha256"]
    )
    assert _sha256(events) == expected["uint8_event_sha256"]


def test_explicit_sc_identity_preserves_inclusive_boundary() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    compatibility = receipt["compatibility_identity"]
    voltage, events = SCInclusivePerfectIntegratorNeuron().simulate_complete(
        protocol["steps"], protocol["current"], backend="python"
    )
    independent_voltage, independent_events = _independent_packet(receipt, inclusive=True)
    np.testing.assert_array_equal(voltage, independent_voltage)
    np.testing.assert_array_equal(events, independent_events)
    assert int(events.sum()) == compatibility["same_protocol_event_count"]
