# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Paired schema and public-contract proof for SC-COMPTE-WM-NETWORK."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any, cast

from sc_neurocore.network import SCCompteWMBehaviorProtocol, SCCompteWMNetworkSpec

SCHEMAS = Path(__file__).parents[1] / "src/sc_neurocore/network/schemas"


def _schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    json_path = SCHEMAS / "sc_compte_wm_network.json"
    toml_path = SCHEMAS / "sc_compte_wm_network.toml"
    encoded = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    with toml_path.open("rb") as handle:
        return encoded, tomllib.load(handle)


def test_sc_network_paired_schemas_are_identical() -> None:
    assert _schemas()[0] == _schemas()[1]


def test_sc_network_schema_matches_public_specification() -> None:
    schema, _ = _schemas()
    spec = SCCompteWMNetworkSpec()
    assert schema["metadata"]["identity"] == spec.identity
    assert schema["metadata"]["specification_version"] == spec.specification_version
    assert schema["topology"]["excitatory_cells"] == spec.n_excitatory
    assert schema["topology"]["inhibitory_cells"] == spec.n_inhibitory
    assert schema["numerics"]["dt_ms"] == spec.dt_ms
    assert schema["external_drive"]["rate_hz"] == spec.external_rate_hz
    assert schema["connectivity"]["ee_conductance_ns"] == spec.recurrent_ee_conductance_ns
    assert schema["connectivity"]["ii_conductance_ns"] == spec.recurrent_ii_conductance_ns


def test_sc_network_schema_enrols_every_runtime_state_array() -> None:
    arrays = _schemas()[0]["state_arrays"]
    assert set(arrays) == {
        "v_exc_mv",
        "v_inh_mv",
        "refractory_exc_ms",
        "refractory_inh_ms",
        "external_ampa_exc",
        "external_ampa_inh",
        "recurrent_nmda",
        "recurrent_nmda_rise",
        "recurrent_gabaa",
    }
    assert sum(item["length"] for item in arrays.values()) == 12_288
    assert {item["dtype"] for item in arrays.values()} == {"binary64"}


def test_sc_network_schema_matches_behavior_protocol() -> None:
    encoded = _schemas()[0]["behavior_protocol"]
    protocol = SCCompteWMBehaviorProtocol()
    cue, distractor, response = protocol.stimuli()
    assert (encoded["duration_ms"], encoded["statistics_window_ms"]) == (
        protocol.duration_ms,
        protocol.window_ms,
    )
    assert encoded["cue"] == {
        "start_ms": cue.start_ms,
        "duration_ms": cue.duration_ms,
        "current_pa": cue.current_pa,
        "center_deg": cue.center_deg,
    }
    assert encoded["distractor"]["center_deg"] == distractor.center_deg
    assert encoded["response"]["kind"] == response.kind


def test_sc_network_schema_states_honest_hardware_boundary() -> None:
    hardware = _schemas()[0]["hardware_boundary"]
    assert hardware["full_network_binary64_equivalence_claimed"] is False
    assert hardware["representative_required"] is True
    assert hardware["yosys_is_device_evidence"] is False
    assert hardware["required_disclosures"] == [
        "state subset",
        "fixed-point format",
        "input surface",
        "latency",
        "error bounds",
    ]
