# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSchemaExport from former test_universal_dsl.py

"""Focused suite: TestSchemaExport from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

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

    def test_to_toml_serializes_bool_and_structured_values(self) -> None:
        toml_str = schema_to_toml(
            {
                "metadata": {"schema_version": 1, "name": "Structured"},
                "extensions": {
                    "enabled": True,
                    "backend_tags": ["python", "verilog"],
                },
            }
        )

        assert "enabled = true" in toml_str
        assert 'backend_tags = ["python", "verilog"]' in toml_str

    def test_schema_property_returns_copy(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        schema1 = neuron.schema
        schema2 = neuron.schema
        assert schema1 == schema2
        assert schema1 is not schema2  # must be a copy
