# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx primary-source and schema identity contracts

"""Primary-source fit and paired-schema contracts for Model 22 AdEx."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_DIRECTORY = REPOSITORY / "src/sc_neurocore/neurons/model_schemas"


def test_source_factory_is_the_published_regular_spiking_fit() -> None:
    """Expose every dimensional value printed in Brette-Gerstner Table 1."""
    neuron = AdExNeuron.brette_gerstner_2005()

    assert neuron.v == neuron.v_rest == neuron.v_reset == -70.6
    assert neuron.w == 0.0
    assert neuron.v_rh == -50.4
    assert neuron.delta_t == 2.0
    assert neuron.tau == pytest.approx(281.0 / 30.0, rel=0.0, abs=1e-15)
    assert neuron.tau_w == 144.0
    assert neuron.a == 4.0
    assert neuron.b == 80.5
    assert neuron.c_m == 281.0
    assert neuron.v_threshold == 20.0


def test_paired_schemas_are_exact_and_separate_source_fit_from_defaults() -> None:
    """Keep TOML and JSON executable/science layers byte-semantically aligned."""
    with (SCHEMA_DIRECTORY / "adex.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((SCHEMA_DIRECTORY / "adex.json").read_text(encoding="utf-8"))

    assert toml_schema == json_schema
    assert toml_schema["metadata"]["schema_version"] == 2
    assert toml_schema["threshold"] == {
        "condition": "v >= v_threshold",
        "detection": "level",
    }
    assert toml_schema["parameters"]["v_threshold"] == -50.0
    published = toml_schema["science"]["published_regular_spiking_parameters"]
    assert published == {
        "C_pF": 281.0,
        "gL_nS": 30.0,
        "EL_mV": -70.6,
        "VT_mV": -50.4,
        "DeltaT_mV": 2.0,
        "tau_w_ms": 144.0,
        "a_nS": 4.0,
        "b_pA": 80.5,
        "Vpeak_mV": 20.0,
        "Vr_mV": -70.6,
    }
    assert "not the fitted parameter set" in toml_schema["extensions"]["runtime_profile"]


@pytest.mark.parametrize("schema_path", ("adex.toml", "adex.json"))
def test_hand_and_schema_profiles_match_complete_maintained_recurrence(schema_path: str) -> None:
    """Prove both schema encodings against the public hand model step by step."""
    hand = AdExNeuron()
    schema = UniversalNeuron.from_schema(SCHEMA_DIRECTORY / schema_path)
    for current in (0.0, 200.0, 500.0):
        hand.reset()
        schema.reset()
        for _ in range(1_000):
            assert schema.step(I=current) == hand.step(current)
            assert schema.state["v"] == pytest.approx(hand.v, rel=0.0, abs=1e-12)
            assert schema.state["w"] == pytest.approx(hand.w, rel=0.0, abs=1e-12)
