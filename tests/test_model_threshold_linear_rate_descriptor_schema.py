# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear rate descriptor and schema tests

"""Descriptor topology and universal-schema hand-model parity."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def test_descriptor_tracks_parameters_and_algebraic_scope() -> None:
    payload = load_descriptor_payload("ThresholdLinearRateNeuron")
    assert payload is not None
    assert set(payload["state"]) == {"r"}
    assert set(payload["parameters"]) == {"theta", "gain"}
    assert payload["integration"] == {"dt": 1.0, "method": "map"}
    assert set(payload["backends"]) == {"python", "rust", "julia", "go", "mojo"}
    assert "no ODE" in payload["dynamics"]["scope"]


def test_schema_map_matches_hand_model() -> None:
    configured = {"theta": 1.5, "gain": 2.0}
    schema = UniversalNeuron.from_schema("threshold_linear_rate", parameter_overrides=configured)
    hand = ThresholdLinearRateNeuron(**configured)
    currents = [1.0, 1.5, 3.0, -4.0]
    schema_trace: list[float] = []
    hand_trace: list[float] = []
    for current in currents:
        schema.step(I=current)
        schema_trace.append(schema.state["r"])
        hand_trace.append(hand.step(current))
    np.testing.assert_array_equal(schema_trace, hand_trace)
