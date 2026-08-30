# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Fourcaud-Trocme ExpIF receipt oracle

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
RECEIPT = REPOSITORY / "src/sc_neurocore/neurons/reference_receipts/expif_fourcaud_trocme_2003.json"


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _independent_receipt_packet() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    v = -65.0
    refractory = 0.0
    dt = 0.01
    voltage_rows: list[float] = []
    refractory_rows: list[float] = []
    event_rows: list[int] = []
    current_rows: list[float] = []

    def rhs(voltage: float, current: float) -> float:
        bounded = min(voltage, -30.0)
        return (-(bounded + 65.0) + 3.48 * math.exp((bounded + 59.9) / 3.48) + current) / 10.0

    for segment in receipt["protocol"]["segments"]:
        current = float(segment["current"])
        for _ in range(int(segment["steps"])):
            event = 0
            if refractory > 0.0:
                refractory = max(0.0, refractory - dt)
                v = -68.0
            else:
                k1 = rhs(v, current)
                k2 = rhs(v + dt * k1, current)
                candidate = v + 0.5 * dt * (k1 + k2)
                if candidate >= -30.0:
                    v = -68.0
                    refractory = 1.7
                    event = 1
                else:
                    v = candidate
            voltage_rows.append(v)
            refractory_rows.append(refractory)
            event_rows.append(event)
            current_rows.append(current)

    return (
        np.asarray(voltage_rows, dtype="<f8"),
        np.asarray(refractory_rows, dtype="<f8"),
        np.asarray(event_rows, dtype="u1"),
        np.asarray(current_rows, dtype="<f8"),
    )


def test_receipt_reproduces_from_an_independent_source_equation_oracle() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt["expected"]
    voltage, refractory, events, currents = _independent_receipt_packet()

    assert len(voltage) == receipt["protocol"]["total_steps"]
    assert int(events.sum()) == expected["spike_count"]
    assert np.flatnonzero(events).tolist() == expected["event_indices"]
    assert voltage[-1] == pytest.approx(expected["final_voltage_mV"], abs=1.0e-12)
    assert refractory[-1] == expected["final_refractory_ms"]
    assert float(voltage.min()) == expected["minimum_voltage_mV"]
    assert float(voltage.max()) == pytest.approx(expected["maximum_voltage_mV"], abs=1.0e-12)
    assert _sha256(voltage) == expected["little_endian_float64_voltage_sha256"]
    assert _sha256(refractory) == expected["little_endian_float64_refractory_sha256"]
    assert _sha256(events) == expected["uint8_event_sha256"]
    assert _sha256(currents) == expected["little_endian_float64_current_sha256"]


def test_analytical_tail_is_derived_from_the_source_exponential_only_flow() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected_tail = 10.0 * math.exp(-(-30.0 - -59.9) / 3.48)
    assert receipt["numerical_specialization"]["analytical_tail_ms"] == pytest.approx(
        expected_tail, abs=1.0e-15
    )
