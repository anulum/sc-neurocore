# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSchemaLoading from former test_universal_dsl.py

"""Focused suite: TestSchemaLoading from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

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

    def test_escape_rate_toml_and_json_are_semantically_identical(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"
        toml = load_schema(schema_dir / "escape_rate.toml")
        json_schema = load_schema(schema_dir / "escape_rate.json")
        assert toml == json_schema
        assert toml["metadata"]["doi"] == "10.1162/089976600300015899"
        assert toml["integration"]["method"] == "exp_euler"
        assert toml["threshold"]["rng_seed"] == 0xACE1

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
