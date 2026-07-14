# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts schema and universal-DSL fidelity

"""Bind the stateless 1943 source rule to schema and RTL emitters."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath
from sc_neurocore.neurons.models.mcculloch_pitts import (
    McCullochPittsNeuron,
    encode_hardware_input,
)
from sc_neurocore.neurons.schema_module_aliases import resolve_schema_join
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema

_REPOSITORY = Path(__file__).resolve().parents[1]


def test_paired_schemas_are_one_stateless_source_contract() -> None:
    """TOML and JSON are equivalent executable interchange forms."""
    source = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
    toml = load_schema(source / "mcculloch_pitts.toml")
    json_schema = load_schema(source / "mcculloch_pitts.json")
    assert toml == json_schema
    assert toml["metadata"]["doi"] == "10.1007/BF02478259"
    assert toml["state"] == {}
    assert toml["dynamics"] == {}
    assert toml["parameters"] == {"theta": 1}
    assert toml["threshold"]["condition"] == "I >= theta"
    assert resolve_schema_join("mcculloch_pitts") == (
        "mcculloch_pitts",
        "McCullochPittsNeuron",
    )


def test_hand_toml_and_json_match_all_encoded_truth_rows() -> None:
    """Separate public inputs and signed schema encoding are exactly equivalent."""
    source = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
    cases = tuple(
        (theta, count, inhibited)
        for theta in (1, 2, 4)
        for count in (0, 1, 2, 4, 17)
        for inhibited in (False, True)
    )
    for schema_path in (source / "mcculloch_pitts.toml", source / "mcculloch_pitts.json"):
        for theta, count, inhibited in cases:
            hand = McCullochPittsNeuron(theta=theta)
            schema = UniversalNeuron.from_schema(
                schema_path,
                parameter_overrides={"theta": float(theta)},
            )
            encoded = encode_hardware_input(count, inhibited)
            assert schema.step(I=float(encoded)) == hand.step(count, inhibited)
            assert schema.state == {}


def test_stateless_schema_level_threshold_repeats_without_history() -> None:
    """No hidden state or crossing latch changes level-triggered logical output."""
    schema = UniversalNeuron.from_schema(
        "mcculloch_pitts",
        parameter_overrides={"theta": 2.0},
    )
    assert [schema.step(I=value) for value in (2.0, 2.0, -1.0, 2.0, 0.0)] == [1, 1, 0, 1, 0]
    assert schema.state == {}
    schema.reset()
    assert schema.state == {}


def test_registered_and_folded_rtl_have_no_invented_cell_state() -> None:
    """Both production emitters expose only the signed input and event output."""
    schema = UniversalNeuron.from_schema("mcculloch_pitts")
    registered = schema.to_verilog(module_name="sc_mcculloch_pitts_contract", fraction=0)
    folded = compile_to_datapath(
        schema.to_equation_neuron(),
        module_name="sc_mcculloch_pitts_folded_contract",
        data_width=32,
        fraction=0,
    )
    for rtl in (registered, folded):
        assert "I_t" in rtl
        assert "spike_out" in rtl
        assert "P_THETA" in rtl
        assert "state_out" not in rtl
        assert "v_next" not in rtl
