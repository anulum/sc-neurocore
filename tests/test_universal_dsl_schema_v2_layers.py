# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSchemaV2Layers from former test_universal_dsl.py

"""Focused suite: TestSchemaV2Layers from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403


class TestSchemaV2Layers:
    """schema_version 2 optional science/validation/provenance/hints layers.

    v2 adds authored, auditable knowledge layers (master plan §4A, invariant I7)
    while keeping every v1 schema a valid v2 schema (backward-compatible).
    """

    @staticmethod
    def _v2_schema() -> dict[str, Any]:
        """Return a minimal but complete schema_version 2 model dictionary."""
        return {
            "metadata": {
                "schema_version": 2,
                "name": "Test v2",
                "maintainers": [
                    {"name": "A. Author", "orcid": "0000-0000-0000-0000", "role": "maintainer"}
                ],
                "citation": "Author et al. 2026",
                "model_version": "1.0.0",
            },
            "state": {"v": 0.0},
            "parameters": {"tau": 10.0},
            "dynamics": {"v": "-v / tau + I"},
            "integration": {"dt": 0.1, "method": "euler"},
            "threshold": {"condition": "v > 1.0", "detection": "level"},
            "reset": {"v": "0.0"},
            "science": {
                "equations_as_published": "dv/dt = -v/tau + I",
                "derivation_note": "identity",
                "assumptions": ["passive membrane"],
                "references": [{"doi": "10.0/x", "note": "primary"}],
            },
            "validation": {
                "model_class": "linear_if",
                "metric": "spike_count_parity",
                "reference": {"kind": "published_figure", "locator": "Fig 1"},
                "tolerance": "0%",
                "audit": {"verified_by": "reviewer", "date": "2026-07-07", "status": "verified"},
            },
            "provenance": {
                "changelog": [
                    {"version": "1.0.0", "date": "2026-07-07", "author": "A", "change": "initial"}
                ],
                "contributors": ["A. Author"],
            },
            "hints": {
                "recommended_precision": "Q16.16",
                "integrator_rationale": "linear, Euler exact",
            },
        }

    def test_v2_schema_loads_and_exposes_layers(self) -> None:
        """A schema_version 2 model loads and exposes every authored layer."""
        neuron = UniversalNeuron.from_dict(self._v2_schema())
        assert neuron.science["equations_as_published"].startswith("dv/dt")
        assert neuron.validation["model_class"] == "linear_if"
        assert neuron.validation["reference"]["kind"] == "published_figure"
        assert neuron.provenance["contributors"] == ["A. Author"]
        assert neuron.hints["recommended_precision"] == "Q16.16"
        # Still simulates through the unchanged core path.
        assert neuron.step(I=50.0) in (0, 1)

    def test_v2_layer_accessors_return_copies(self) -> None:
        """The layer accessors return copies, so callers cannot mutate internals."""
        neuron = UniversalNeuron.from_dict(self._v2_schema())
        neuron.science["equations_as_published"] = "MUTATED"
        assert neuron.science["equations_as_published"].startswith("dv/dt")

    def test_v1_schema_has_empty_v2_layers(self) -> None:
        """Every bundled v1 schema loads with empty authored layers (backward-compatible)."""
        neuron = UniversalNeuron.from_schema("lif")
        assert neuron.science == {}
        assert neuron.validation == {}
        assert neuron.provenance == {}
        assert neuron.hints == {}

    def test_load_schema_accepts_v2_rejects_unknown(self, tmp_path: Path) -> None:
        """load_schema accepts version 2 and rejects an unsupported version."""
        from sc_neurocore.neurons.universal_dsl import _SUPPORTED_SCHEMA_VERSIONS

        assert 2 in _SUPPORTED_SCHEMA_VERSIONS
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "metadata": {"schema_version": 99},
                    "state": {"v": 0.0},
                    "dynamics": {"v": "I"},
                    "integration": {"dt": 0.1, "method": "euler"},
                }
            )
        )
        with pytest.raises(ValueError, match="not supported"):
            load_schema(bad)
