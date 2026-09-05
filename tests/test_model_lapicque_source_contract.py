# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque source identity and paired-schema contracts

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

from sc_neurocore.neurons.models import LapicqueNeuron, SCLapicqueLIFNeuron
from sc_neurocore.network.population import Population


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "src/sc_neurocore/neurons/model_schemas"


def test_paired_source_and_sc_schemas_are_exact_and_disjoint() -> None:
    for stem in ("lapicque", "sc_lapicque_lif"):
        with (SCHEMAS / f"{stem}.toml").open("rb") as handle:
            toml_schema = tomllib.load(handle)
        json_schema = json.loads((SCHEMAS / f"{stem}.json").read_text(encoding="utf-8"))
        assert toml_schema == json_schema

    source = json.loads((SCHEMAS / "lapicque.json").read_text(encoding="utf-8"))
    compatibility = json.loads((SCHEMAS / "sc_lapicque_lif.json").read_text(encoding="utf-8"))
    assert source["metadata"]["name"] == "LapicqueNeuron"
    assert source["extensions"]["event_contract"].endswith("no automatic source reset")
    assert "excited" in source["state"]
    assert "event_flag" not in source["state"]
    assert source["reset"] == {"v": "v", "excited": "1.0"}
    assert compatibility["metadata"]["name"] == "SCLapicqueLIFNeuron"
    assert compatibility["reset"] == {"v": "v_reset"}


def test_source_factory_preserves_lapicques_distinct_circuit_parameters() -> None:
    neuron = LapicqueNeuron.lapicque_1907()
    assert neuron.profile == "lapicque_1907"
    assert neuron.source_beta == 1.0
    assert neuron.source_alpha == 11.0
    assert neuron.excited is False
    assert neuron.dt == 0.01


@pytest.mark.parametrize("duration", (0.1, 0.5, 1.0, 2.0, 5.0))
def test_strength_duration_voltage_reaches_threshold_analytically(duration: float) -> None:
    neuron = LapicqueNeuron.lapicque_1907()
    source_voltage = neuron.source_threshold_voltage(duration)
    reached = (
        source_voltage
        * neuron.polarization_resistance
        / (neuron.series_resistance + neuron.polarization_resistance)
        * (1.0 - math.exp(-duration / neuron.source_beta))
    )
    assert reached == pytest.approx(neuron.v_threshold, abs=2.0e-15)


def test_source_event_latches_without_implicit_reset_or_repetition() -> None:
    neuron = LapicqueNeuron.lapicque_1907()
    voltage, events = neuron.simulate_complete(2_000, 22.0, backend="python")
    assert events.sum() == 1
    assert events.nonzero()[0].tolist() == [69]
    assert voltage[69] >= neuron.v_threshold
    assert voltage[-1] > voltage[69]
    assert neuron.excited is True

    continued_voltage, continued_events = neuron.simulate_complete(100, 22.0, backend="python")
    assert continued_events.sum() == 0
    assert continued_voltage[-1] > voltage[-1]


def test_sc_compatibility_identity_is_explicit_and_repetitively_resettable() -> None:
    neuron = SCLapicqueLIFNeuron()
    assert neuron.profile == "sc_lif"
    _, events = neuron.simulate_complete(1_000, 5.0, backend="python")
    assert int(events.sum()) == 200
    assert neuron.excited is False


def test_population_canonical_string_routes_source_and_sc_parameters_remain_compatible() -> None:
    source = Population("LapicqueNeuron", 2)
    assert [neuron.profile for neuron in source.neurons] == ["lapicque_1907"] * 2

    compatibility = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 0.5})
    assert [neuron.profile for neuron in compatibility.neurons] == ["sc_lif"] * 2
    assert [neuron.tau for neuron in compatibility.neurons] == [5.0] * 2

    explicit = Population("SCLapicqueLIFNeuron", 2)
    assert [neuron.profile for neuron in explicit.neurons] == ["sc_lif"] * 2
