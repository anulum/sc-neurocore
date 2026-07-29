# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Amari 1977 finite-grid oracle

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.accel.amari_field import simulate_amari_field
from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "src/sc_neurocore/neurons/reference_trace_data/amari_field_doi.json"
SCHEMAS = ROOT / "src/sc_neurocore/neurons/model_schemas"


def _iterate(receipt: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    """Evaluate equation (3) without importing any production recurrence."""
    config = receipt["configuration"]
    n = int(config["n"])
    state = [float(value) for value in receipt["initial_state"]]
    kernel = []
    for offset in range(n):
        distance = min(offset, n - offset) * float(config["dx"])
        kernel.append(
            float(config["a_exc"]) * math.exp(-float(config["a_width"]) * distance)
            - float(config["b_inh"]) * math.exp(-float(config["b_width"]) * distance)
        )
    states: list[list[float]] = []
    rates: list[float] = []
    for drive in receipt["currents"]:
        candidate = []
        for site in range(n):
            interaction = sum(
                kernel[(site - source) % n] for source, value in enumerate(state) if value > 0.0
            )
            candidate.append(
                state[site]
                + (-state[site] + interaction * float(config["dx"]) + float(drive[site]))
                * (float(config["dt"]) / float(config["tau"]))
            )
        state = candidate
        states.append(list(state))
        rates.append(sum(value > 0.0 for value in state) / n)
    return states, rates


def test_primary_equation_oracle_matches_frozen_receipt() -> None:
    receipt = json.loads(TRACE.read_text(encoding="utf-8"))
    assert receipt["doi"] == "10.1007/BF00337259"
    states, rates = _iterate(receipt)
    np.testing.assert_allclose(states, receipt["states"], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(rates, receipt["mean_rates"])
    np.testing.assert_allclose(states[-1], receipt["final_state"], rtol=0.0, atol=1.0e-15)


def test_python_batch_matches_primary_equation_receipt() -> None:
    receipt = json.loads(TRACE.read_text(encoding="utf-8"))
    result = simulate_amari_field(
        receipt["initial_state"],
        **{k: v for k, v in receipt["configuration"].items() if k != "n"},
        currents=receipt["currents"],
        backend="python",
    )
    np.testing.assert_allclose(result["states"], receipt["states"], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(result["mean_rates"], receipt["mean_rates"])


def test_paired_schemas_are_exact_and_execute_the_four_site_specialization() -> None:
    toml_schema = load_schema(SCHEMAS / "amari_field.toml")
    json_schema = load_schema(SCHEMAS / "amari_field.json")
    assert toml_schema == json_schema
    schemas = [
        UniversalNeuron.from_schema(SCHEMAS / "amari_field.toml"),
        UniversalNeuron.from_schema(SCHEMAS / "amari_field.json"),
    ]
    hand = AmariNeuralField(n=4)
    for current in (0.1, -0.2, 0.3, 0.05, -0.1):
        hand.step(current)
        state = hand.u
        assert state is not None
        for schema in schemas:
            assert schema.step(I=current) == 0
            np.testing.assert_allclose(
                np.asarray(list(schema.state.values()), dtype=np.float64),
                state,
                rtol=0.0,
                atol=1.0e-15,
            )
