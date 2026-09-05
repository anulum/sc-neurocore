# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema export preservation and replay

"""Schema field preservation and executable model replay after TOML export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.neurons.universal_dsl import (
    UniversalNeuron,
    load_schema,
    schema_to_toml,
)


class TestSchemaExport:
    """Test JSON and TOML export."""

    def test_to_json_roundtrip(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        exported = neuron.to_json()
        parsed = json.loads(exported)
        assert parsed["metadata"]["name"] == "LIF"
        assert parsed["dynamics"]["v"] == "-(v - v_rest) / tau_m + R * I / C"

    def test_to_toml_contains_sections(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        toml_str = neuron.to_toml()
        assert "[metadata]" in toml_str
        assert "[dynamics]" in toml_str
        assert "[threshold]" in toml_str

    def test_to_toml_serializes_bool_and_structured_values(self, tmp_path: Path) -> None:
        schema = load_schema("lif")
        schema["extensions"] = {"enabled": True, "backend_tags": ["python", "verilog"]}
        exported = tmp_path / "structured.toml"
        exported.write_text(schema_to_toml(schema), encoding="utf-8")
        assert load_schema(exported) == schema

    def test_schema_property_returns_copy(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        schema1 = neuron.schema
        schema2 = neuron.schema
        assert schema1 == schema2
        assert schema1 is not schema2  # must be a copy


@pytest.mark.parametrize(
    "source_path",
    sorted(
        path
        for path in (
            Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        ).iterdir()
        if path.suffix in {".toml", ".json"}
    ),
    ids=lambda path: path.name,
)
def test_bundled_schema_roundtrip_preserves_every_section(
    source_path: Path, tmp_path: Path
) -> None:
    """Public export and loading retain the complete authored catalogue record."""
    source = load_schema(source_path)
    exported = tmp_path / "model.toml"
    exported.write_text(schema_to_toml(source), encoding="utf-8")
    assert load_schema(exported) == source


@pytest.mark.parametrize(
    ("name", "drive"),
    [
        ("lapicque", 22.0),
        ("lif", 20.0),
        ("fitzhugh_nagumo", 1.0),
        ("poisson", 1000.0),
        ("sc_perfect_integrator", 1.0),
    ],
)
def test_exported_model_replays_state_and_events(name: str, drive: float, tmp_path: Path) -> None:
    """Reloaded deterministic and seeded stochastic models reproduce execution."""
    original = UniversalNeuron.from_schema(name)
    exported = tmp_path / "replay.toml"
    exported.write_text(original.to_toml(), encoding="utf-8")
    restored = UniversalNeuron.from_schema(exported)
    assert restored.schema == original.schema
    events = []
    for current in [0.0] * 10 + [drive] * 200 + [0.0] * 10:
        observed = restored.step(I=current)
        assert observed == original.step(I=current)
        assert restored.state == original.state
        events.append(observed)
    assert 0 < sum(events) < len(events)
    restored.reset()
    original.reset()
    assert restored.state == original.state


def test_authoring_metadata_nested_values_and_escaping_roundtrip(tmp_path: Path) -> None:
    """A real model retains rich authoring fields, special keys and empty tables."""
    source = load_schema("lapicque")
    source["extensions"] = {
        "annotation": 'Voltage "v"\npath \\ trace\tŠotek',
        "coordinate.mapping": {"units": "mV", "scale": 1.0},
        "reviewers": [{"name": "A", "approved": False}, {"name": "B", "approved": True}],
        "empty": {},
        "labels": [],
    }
    original = UniversalNeuron.from_dict(source)
    exported = tmp_path / "annotated.toml"
    exported.write_text(original.to_toml(), encoding="utf-8")
    assert load_schema(exported) == source
    assert original.schema == source


def test_unsupported_metadata_fails_without_mutating_source() -> None:
    """Unrepresentable metadata is rejected rather than silently discarded."""
    source = load_schema("lapicque")
    source["extensions"] = {"missing": None}
    before = json.dumps(source, sort_keys=True)
    with pytest.raises(TypeError):
        schema_to_toml(source)
    assert json.dumps(source, sort_keys=True) == before
