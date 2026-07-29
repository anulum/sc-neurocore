# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Compte primary-equation receipt

"""Recompute the source-bounded pyramidal/channel recurrence independently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

_RECEIPT = (
    Path(__file__).parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/compte_wm_2000_pyramidal.json"
)


def _oracle() -> tuple[list[tuple[float, ...]], list[int]]:
    v, s_ampa, s_nmda, x_nmda, s_gaba, refractory = -70.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = 0.02
    rows: list[tuple[float, ...]] = []
    event_indices: list[int] = []
    for index in range(1024):
        current = 1.0 + 0.2 * math.sin(index * 0.03)
        if index % 11 == 0:
            s_ampa += 1.0
        if index % 17 == 0:
            x_nmda += 1.0
        if index % 23 == 0:
            s_gaba += 1.0
        active = refractory <= 0.0

        def derivative(state: tuple[float, ...]) -> tuple[float, ...]:
            voltage, ampa, nmda, precursor, gaba = state
            d_v = 0.0
            if active:
                block = 1.0 / (1.0 + math.exp(-0.062 * voltage) / 3.57)
                d_v = (
                    -0.025 * (voltage + 70.0)
                    - 0.0031 * ampa * voltage
                    - 0.000381 * block * nmda * voltage
                    - 0.001336 * gaba * (voltage + 70.0)
                    + current
                ) / 0.5
            return (
                d_v,
                -ampa / 2.0,
                -nmda / 100.0 + 0.5 * precursor * (1.0 - nmda),
                -precursor / 2.0,
                -gaba / 10.0,
            )

        initial = (v, s_ampa, s_nmda, x_nmda, s_gaba)
        first = derivative(initial)
        midpoint = tuple(
            value + 0.5 * dt * slope for value, slope in zip(initial, first, strict=True)
        )
        second = derivative(midpoint)
        v, s_ampa, s_nmda, x_nmda, s_gaba = tuple(
            value + dt * slope for value, slope in zip(initial, second, strict=True)
        )
        event = 0
        refractory = max(0.0, refractory - dt)
        if not active:
            v = -60.0
        elif v >= -50.0:
            v, refractory, event = -60.0, 2.0, 1
            event_indices.append(index)
        rows.append((v, s_ampa, s_nmda, x_nmda, s_gaba, refractory, float(event)))
    return rows, event_indices


def test_primary_equation_receipt_is_reproducible() -> None:
    receipt = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    rows, indices = _oracle()
    payload = b"".join(struct.pack("<ddddddd", *row) for row in rows)
    assert hashlib.sha256(payload).hexdigest() == receipt["oracle"]["trace_sha256"]
    assert indices == receipt["oracle"]["event_indices"]
    assert rows[-1][:6] == pytest.approx(receipt["oracle"]["final_state"], abs=0.0)


def test_hand_model_matches_every_independent_observable() -> None:
    expected, _ = _oracle()
    neuron = CompteWMNeuron()
    actual = []
    for index in range(1024):
        event = neuron.step(
            1.0 + 0.2 * math.sin(index * 0.03),
            index % 17 == 0,
            external_spike=index % 11 == 0,
            inhibitory_spike=index % 23 == 0,
        )
        actual.append((*neuron.get_state().values(), float(event)))
    assert actual == expected
