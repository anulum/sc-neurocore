# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Lapicque 1907 source-receipt oracle

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "src/sc_neurocore/neurons/reference_receipts/lapicque_1907.json"


def _sha256(array: npt.NDArray[np.generic]) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _independent_packet(
    receipt: dict[str, object],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    numerical = receipt["numerical_specialization"]
    protocol = receipt["protocol"]
    assert isinstance(numerical, dict)
    assert isinstance(protocol, dict)
    threshold = float(numerical["v_threshold"])
    capacitance = float(numerical["capacitance_K"])
    series = float(numerical["series_resistance_R"])
    polarization = float(numerical["polarization_resistance_rho"])
    dt = float(numerical["dt_ms"])
    source_voltage = float(protocol["source_voltage_V"])
    steps = int(protocol["steps"])
    beta = capacitance * series * polarization / (series + polarization)
    v_inf = source_voltage * polarization / (series + polarization)
    decay = math.exp(-dt / beta)
    voltage = np.empty(steps, dtype="<f8")
    events = np.zeros(steps, dtype="u1")
    v = float(numerical["v_initial"])
    excited = False
    for index in range(steps):
        v = v_inf + (v - v_inf) * decay
        event = not excited and v >= threshold
        events[index] = event
        excited = excited or event
        voltage[index] = v
    return voltage, events


def test_receipt_reproduces_from_the_independent_source_equation() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt["expected"]
    voltage, events = _independent_packet(receipt)
    assert int(events.sum()) == expected["event_count"]
    assert np.flatnonzero(events).tolist() == expected["event_indices_zero_based"]
    assert voltage[-1] == expected["final_polarization"]
    assert _sha256(voltage) == expected["little_endian_float64_polarization_sha256"]
    assert _sha256(events) == expected["uint8_event_sha256"]


def test_receipt_matches_production_complete_packet() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    expected = receipt["expected"]
    neuron = LapicqueNeuron.lapicque_1907()
    voltage, events = neuron.simulate_complete(
        protocol["steps"], protocol["source_voltage_V"], backend="python"
    )
    assert (
        _sha256(voltage.astype("<f8", copy=False))
        == expected["little_endian_float64_polarization_sha256"]
    )
    assert _sha256(events) == expected["uint8_event_sha256"]


def test_receipt_strength_duration_points_are_rederived_not_copied() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    numerical = receipt["numerical_specialization"]
    alpha = (
        numerical["v_threshold"]
        * (numerical["series_resistance_R"] + numerical["polarization_resistance_rho"])
        / numerical["polarization_resistance_rho"]
    )
    beta = numerical["beta_ms"]
    for point in receipt["strength_duration_points"]:
        expected = alpha / -math.expm1(-point["duration_ms"] / beta)
        assert point["threshold_source_voltage"] == pytest.approx(expected, abs=1.0e-14)
