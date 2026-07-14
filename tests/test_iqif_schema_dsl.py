# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF schema and universal-DSL fidelity

"""TOML/JSON parity and exact hand/schema iteration for IQIF."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.neurons.schema_module_aliases import class_for_schema, resolve_schema_join
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DYNAMICS = (
    "max(v_min, v + (((a * (v_rest - v)) // 8 if v < branch_point "
    "else (b * (v - v_threshold)) // 8) + I))"
)


def test_toml_and_json_payloads_are_identical() -> None:
    """Both serialisations define one source-bound executable contract."""
    toml = load_schema(_SCHEMA_DIR / "iqif.toml")
    json_schema = load_schema(_SCHEMA_DIR / "iqif.json")
    assert toml == json_schema
    assert toml["metadata"]["doi"] == "10.1109/AICAS51828.2021.9458572"
    assert toml["state"] == {"v": 128}
    assert toml["parameters"]["branch_point"] == 164
    assert toml["integration"] == {"dt": 1.0, "method": "map"}
    assert toml["dynamics"]["v"] == _DYNAMICS
    assert toml["threshold"] == {"condition": "v > v_max", "detection": "level"}
    assert toml["reset"] == {"v": "v_reset"}


def test_schema_alias_resolves_the_public_hand_class() -> None:
    """Readiness, Studio and co-sim tooling share the exact IQIF join."""
    assert class_for_schema("iqif") == "IntegerQIFNeuron"
    assert resolve_schema_join("iqif") == ("iqif", "IntegerQIFNeuron")


def test_hand_toml_and_json_match_every_source_tutorial_tick() -> None:
    """The executable schema preserves all 400 events and integer states."""
    hand = IntegerQIFNeuron()
    toml = UniversalNeuron.from_schema(_SCHEMA_DIR / "iqif.toml")
    json_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "iqif.json")
    hand_trace: list[int] = []
    hand_spikes: list[int] = []
    for index in range(400):
        hand_event = hand.step(10)
        toml_event = toml.step(I=10)
        json_event = json_schema.step(I=10)
        assert toml_event == json_event == hand_event
        assert toml.state == json_schema.state == {"v": float(hand.v)}
        hand_trace.append(hand.v)
        if hand_event:
            hand_spikes.append(index)
    assert hand_spikes == list(range(14, 400, 15))
    assert hand_trace[-1] == 198


def test_configured_schema_override_requires_and_uses_recomputed_branch_point() -> None:
    """A non-default parameter profile remains exact when its derived field is supplied."""
    hand = IntegerQIFNeuron(
        v=100,
        v_rest=96,
        v_threshold=180,
        v_reset=120,
        a=3,
        b=5,
        v_max=240,
        v_min=4,
    )
    schema = UniversalNeuron.from_schema(
        "iqif",
        parameter_overrides={
            "v_rest": 96,
            "v_threshold": 180,
            "v_reset": 120,
            "a": 3,
            "b": 5,
            "v_max": 240,
            "v_min": 4,
            "branch_point": hand.branch_point,
        },
    )
    schema.to_equation_neuron().state["v"] = 100.0
    for _ in range(128):
        assert schema.step(I=17) == hand.step(17)
        assert schema.state["v"] == float(hand.v)


def test_schema_strict_upper_boundary_matches_hand_model() -> None:
    """The level threshold observes the candidate before applying v_reset."""
    schema = UniversalNeuron.from_schema("iqif")
    equation = schema.to_equation_neuron()
    equation.state["v"] = 255.0
    assert schema.step(I=-6) == 0
    assert schema.state["v"] == 255.0
    assert schema.step(I=-5) == 1
    assert schema.state["v"] == 128.0
