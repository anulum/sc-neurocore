# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Brunel-Wang primary-equation receipt

"""Recompute the Methods 2.2--2.3 pyramidal-cell specialization independently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron

_RECEIPT = (
    Path(__file__).parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/brunel_wang_2001_pyramidal.json"
)


def _gates(index: int) -> tuple[float, float, float, float]:
    return (
        0.035 + 0.018 * (1.0 + math.sin(index * 0.071)),
        0.12 + 0.05 * (1.0 + math.cos(index * 0.053)),
        0.08 + 0.04 * (1.0 + math.sin(index * 0.037 + 0.2)),
        0.03 + 0.02 * (1.0 + math.cos(index * 0.089)),
    )


def _oracle() -> tuple[list[tuple[float, float, float]], list[int]]:
    v, refractory, dt = -70.0, 0.0, 0.1
    rows: list[tuple[float, float, float]] = []
    event_indices: list[int] = []
    for index in range(256):
        ext, ampa, nmda, gaba = _gates(index)
        if refractory > 0.0:
            v, refractory, event = -55.0, max(0.0, refractory - dt), 0
        else:

            def derivative(voltage: float) -> float:
                block = 1.0 / (1.0 + math.exp(-0.062 * voltage) / 3.57)
                current = -2.08 * voltage * ext - 0.104 * voltage * ampa
                current -= 0.327 * block * voltage * nmda
                current -= 1.25 * (voltage + 70.0) * gaba
                return -(voltage + 70.0) / 20.0 + current / 0.5

            k1 = derivative(v)
            midpoint = v + 0.5 * dt * k1
            candidate = v + dt * derivative(midpoint)
            if candidate >= -50.0:
                v, refractory, event = -55.0, 2.0, 1
            else:
                v, event = candidate, 0
        if event:
            event_indices.append(index)
        rows.append((v, refractory, float(event)))
    return rows, event_indices


def test_primary_equation_receipt_is_reproducible() -> None:
    """Pin the independent trace hash, events, and final state."""
    receipt = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    rows, indices = _oracle()
    payload = b"".join(struct.pack("<ddd", *row) for row in rows)
    assert hashlib.sha256(payload).hexdigest() == receipt["oracle"]["trace_sha256"]
    assert indices == receipt["oracle"]["event_indices"]
    assert rows[-1][:2] == pytest.approx(
        (receipt["oracle"]["final_voltage_mv"], receipt["oracle"]["final_refractory_ms"]),
        abs=0.0,
    )


def test_hand_model_matches_independent_primary_equation() -> None:
    """Keep the maintained class bound to every independent observable."""
    expected, _ = _oracle()
    neuron = BrunelWangNeuron()
    actual = []
    for index in range(256):
        event = neuron.step(*_gates(index))
        actual.append((neuron.v, neuron.get_state()["ref_remaining"], float(event)))
    assert actual == expected
