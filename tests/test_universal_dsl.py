# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Universal Neuron DSL

"""Test suite for the Universal Neuron DSL.

Covers:
- TOML and JSON schema loading
- Bare-name resolution against bundled schemas
- Simulation parity with hand-crafted model classes
- Parameter overrides and integration method switching
- Schema export (to_json, to_toml)
- Error handling for invalid/missing schemas
- Forward-compatible extension fields
- Schema version gating
- Introspection methods
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

from sc_neurocore.neurons.universal_dsl import (
    UniversalNeuron,
    list_bundled_schemas,
    load_schema,
    schema_to_toml,
)


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


class TestSchemaLoading:
    """Test TOML and JSON schema loading from bundled schemas."""

    def test_load_lif_toml(self) -> None:
        schema = load_schema("lif")
        assert schema["metadata"]["name"] == "LIF"
        assert schema["metadata"]["schema_version"] == 1
        assert "v" in schema["dynamics"]

    def test_load_lif_json(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"
        schema = load_schema(schema_dir / "lif.json")
        assert schema["metadata"]["name"] == "LIF"

    def test_load_fitzhugh_nagumo(self) -> None:
        schema = load_schema("fitzhugh_nagumo")
        assert schema["metadata"]["year"] == 1961
        assert "v" in schema["dynamics"]
        assert "w" in schema["dynamics"]

    def test_load_izhikevich(self) -> None:
        schema = load_schema("izhikevich")
        assert schema["metadata"]["year"] == 2003
        assert "v" in schema["state"]
        assert "u" in schema["state"]

    def test_load_hindmarsh_rose(self) -> None:
        schema = load_schema("hindmarsh_rose")
        assert schema["metadata"]["year"] == 1984
        assert len(schema["state"]) == 3

    def test_load_adex(self) -> None:
        schema = load_schema("adex")
        assert schema["metadata"]["year"] == 2005
        assert "delta_T" in schema["parameters"]

    def test_nonexistent_schema_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_model_xyz")

    def test_unsupported_format_raises(self) -> None:
        # Create a temp file with unsupported extension
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: true")
            path = f.name
        with pytest.raises(ValueError, match="Unsupported schema format"):
            load_schema(path)
        Path(path).unlink()

    def test_explicit_missing_schema_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            load_schema(tmp_path / "missing.json")

    def test_toml_loader_uses_tomli_on_python_pre_311(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        schema_path = tmp_path / "fallback.toml"
        schema_path.write_text(
            "\n".join(
                (
                    "[metadata]",
                    "schema_version = 1",
                    'name = "Fallback"',
                    "[state]",
                    "v = 0.0",
                    "[dynamics]",
                    'v = "I"',
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "version_info", (3, 10))

        schema = load_schema(schema_path)

        assert schema["metadata"]["name"] == "Fallback"


class TestListBundledSchemas:
    """Test discovery of bundled schemas."""

    def test_lists_all_bundled(self) -> None:
        names = list_bundled_schemas()
        assert "lif" in names
        assert "fitzhugh_nagumo" in names
        assert "izhikevich" in names
        assert "hindmarsh_rose" in names
        assert "adex" in names

    def test_returns_sorted(self) -> None:
        names = list_bundled_schemas()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Instantiation and simulation
# ---------------------------------------------------------------------------


class TestUniversalNeuronSimulation:
    """Test that UniversalNeuron produces physically reasonable dynamics."""

    def test_lif_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        spikes = sum(neuron.step(I=30.0) for _ in range(200))
        assert spikes > 0, "LIF should spike with strong input"

    def test_lif_no_spike_without_input(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        spikes = sum(neuron.step(I=0.0) for _ in range(200))
        assert spikes == 0, "LIF should not spike without input"

    def test_fitzhugh_nagumo_oscillates(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        spikes = sum(neuron.step(I=0.5) for _ in range(2000))
        assert spikes > 0, "FHN should oscillate with I=0.5"

    def test_izhikevich_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        spikes = sum(neuron.step(I=10.0) for _ in range(200))
        assert spikes > 0, "Izhikevich should spike with I=10"

    def test_hindmarsh_rose_evolves(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        initial_x = neuron.state["x"]
        for _ in range(500):
            neuron.step(I=3.0)
        assert neuron.state["x"] != initial_x, "HR should evolve from initial state"

    def test_adex_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("adex")
        spikes = sum(neuron.step(I=500.0) for _ in range(500))
        assert spikes > 0, "AdEx should spike with strong current"


class TestParameterOverrides:
    """Test runtime parameter overrides."""

    def test_override_tau(self) -> None:
        # Faster membrane time constant → more spikes
        slow = UniversalNeuron.from_schema("lif")
        fast = UniversalNeuron.from_schema("lif", parameter_overrides={"tau_m": 2.0})
        slow_spikes = sum(slow.step(I=20.0) for _ in range(200))
        fast_spikes = sum(fast.step(I=20.0) for _ in range(200))
        assert fast_spikes >= slow_spikes

    def test_override_dt(self) -> None:
        neuron = UniversalNeuron.from_schema("lif", dt_override=0.5)
        spikes = sum(neuron.step(I=50.0) for _ in range(400))
        # With dt=0.5 (half the default 1.0), the neuron should still spike
        assert spikes > 0, "LIF with dt_override should still produce spikes"


class TestResetAndState:
    """Test reset and state introspection."""

    def test_reset_restores_initial(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        initial = dict(neuron.state)
        for _ in range(100):
            neuron.step(I=30.0)
        neuron.reset()
        assert neuron.state == initial

    def test_state_is_dict(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        assert isinstance(neuron.state, dict)
        assert "v" in neuron.state
        assert "w" in neuron.state


# ---------------------------------------------------------------------------
# JSON / TOML parity
# ---------------------------------------------------------------------------


class TestFormatParity:
    """Verify that TOML and JSON schemas produce identical simulation results."""

    def test_lif_toml_vs_json_parity(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"

        toml_neuron = UniversalNeuron.from_schema(schema_dir / "lif.toml")
        json_neuron = UniversalNeuron.from_schema(schema_dir / "lif.json")

        for _ in range(100):
            s1 = toml_neuron.step(I=20.0)
            s2 = json_neuron.step(I=20.0)
            assert s1 == s2

        assert toml_neuron.state["v"] == json_neuron.state["v"]

    def test_izhikevich_toml_vs_json_parity(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"

        toml_neuron = UniversalNeuron.from_schema(schema_dir / "izhikevich.toml")
        json_neuron = UniversalNeuron.from_schema(schema_dir / "izhikevich.json")

        for _ in range(100):
            s1 = toml_neuron.step(I=10.0)
            s2 = json_neuron.step(I=10.0)
            assert s1 == s2

        for var in ("v", "u"):
            assert toml_neuron.state[var] == json_neuron.state[var]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestIntrospection:
    """Test introspection methods."""

    def test_name(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        assert neuron.name == "FitzHugh-Nagumo"

    def test_doi(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        assert neuron.doi == "10.1109/TNN.2003.820440"

    def test_list_state_variables(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        assert set(neuron.list_state_variables()) == {"x", "y", "z"}

    def test_list_parameters(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        params = neuron.list_parameters()
        assert "b" in params
        assert "r" in params
        assert "s" in params

    def test_list_equations(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        eqs = neuron.list_equations()
        assert "v" in eqs
        assert "w" in eqs
        assert "v**3" in eqs["v"]

    def test_repr(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        r = repr(neuron)
        assert "UniversalNeuron" in r
        assert "LIF" in r

    def test_extensions_property(self) -> None:
        neuron = UniversalNeuron.from_schema("adex")
        ext = neuron.extensions
        assert "integrator_options" in ext

    def test_to_equation_neuron(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = UniversalNeuron.from_schema("lif")
        eq_neuron = neuron.to_equation_neuron()
        assert isinstance(eq_neuron, EquationNeuron)

    def test_to_verilog_sanitizes_default_module_name(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_compile_to_verilog(
            neuron: object,
            *,
            module_name: str,
            **kwargs: Any,
        ) -> str:
            captured["neuron"] = neuron
            captured["module_name"] = module_name
            captured["kwargs"] = kwargs
            return f"module {module_name}; endmodule"

        monkeypatch.setattr(
            "sc_neurocore.compiler.equation_compiler.compile_to_verilog",
            fake_compile_to_verilog,
        )
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")

        verilog = neuron.to_verilog(data_width=12)

        assert verilog == "module sc_fitzhugh_nagumo; endmodule"
        assert captured["neuron"] is neuron.to_equation_neuron()
        assert captured["module_name"] == "sc_fitzhugh_nagumo"
        assert captured["kwargs"] == {"data_width": 12}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling for invalid schemas."""

    def test_empty_dynamics_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one ODE"):
            UniversalNeuron.from_dict(
                {
                    "metadata": {"schema_version": 1, "name": "Empty"},
                    "state": {"v": 0.0},
                    "dynamics": {},
                }
            )

    def test_unsupported_version_in_file(self) -> None:
        """Version gate fires when loading a schema with unsupported version."""
        import tempfile

        bad_schema = {
            "metadata": {"schema_version": 999, "name": "Future"},
            "state": {"v": 0.0},
            "dynamics": {"v": "I"},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(bad_schema, f)
            path = f.name
        with pytest.raises(ValueError, match="Schema version.*not supported"):
            load_schema(path)
        Path(path).unlink()

    def test_from_dict_works(self) -> None:
        neuron = UniversalNeuron.from_dict(
            {
                "metadata": {"schema_version": 1, "name": "TestModel"},
                "state": {"v": 0.0},
                "parameters": {},
                "dynamics": {"v": "I"},
                "integration": {"dt": 0.1, "method": "euler"},
            }
        )
        neuron.step(I=1.0)
        assert neuron.state["v"] != 0.0


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
