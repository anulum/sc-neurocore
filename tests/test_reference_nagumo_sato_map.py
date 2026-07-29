# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Nagumo–Sato equation oracle

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sc_neurocore.accel.nagumo_sato_map import simulate_nagumo_sato_map
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

_TRACE = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/nagumo_sato_map_primary.json"
)


def test_independent_source_equation_matches_committed_trace() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    config = receipt["configuration"]
    y = float(config["y"])
    expected_y: list[float] = []
    expected_events: list[int] = []
    for current in receipt["current"]:
        y = config["k"] * y - config["alpha"] * int(y >= 0.0) + config["bias"] + current
        expected_y.append(y)
        expected_events.append(int(y >= 0.0))
    np.testing.assert_allclose(expected_y, receipt["y"], rtol=0.0, atol=1e-15)
    assert expected_events == receipt["x"] == receipt["spikes"]


def test_hand_model_and_toml_schema_match_source_oracle() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    config = receipt["configuration"]
    result = simulate_nagumo_sato_map(**config, current=receipt["current"], backend="python")
    np.testing.assert_allclose(result["y"], receipt["y"], rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(result["spikes"], receipt["spikes"])
    schema = UniversalNeuron.from_schema("nagumo_sato_map")
    observed = []
    events = []
    for current in receipt["current"]:
        events.append(int(bool(schema.step(I=current))))
        observed.append(schema.state["y"])
    np.testing.assert_allclose(observed, receipt["y"], rtol=0.0, atol=1e-15)
    assert events == receipt["spikes"]
