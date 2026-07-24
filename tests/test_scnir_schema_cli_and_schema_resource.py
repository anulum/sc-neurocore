# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cli_and_schema_resource) from former test_scnir_schema.py

from __future__ import annotations

from tests.scnir_schema_support import *  # noqa: F403


def test_scnir_json_schema_resource_is_bundled() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas/scnir/scnir.schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["$id"].endswith("/schemas/scnir/scnir.schema.json")
    assert payload["properties"]["schema_version"]["const"] == SCNIR_SCHEMA_VERSION
    assert "bitstream_length" in json.dumps(payload)
    assert "delay_steps" in json.dumps(payload)
    assert "signal_kind" in json.dumps(payload)
    assert "transforms" in json.dumps(payload)
    assert "hierarchy" in json.dumps(payload)
    assert "module_name" in json.dumps(payload)
    assert "correlation_constraints" in json.dumps(payload)
    assert "online_learning" in json.dumps(payload)


def test_scnir_validate_cli_reports_valid_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "valid.scnir.json"
    write_scnir(path, _valid_document())

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "validate", str(path)]):
        rc = main()

    assert rc == 0
    assert "SC-NIR valid" in capsys.readouterr().out


def test_scnir_validate_cli_reports_invalid_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.scnir.json"
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["source"]["seed"] = -1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "validate", str(path)]):
        rc = main()

    assert rc == 1
    assert "seed" in capsys.readouterr().out


def test_scnir_upgrade_cli_writes_canonical_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.scnir.json"
    output_path = tmp_path / "upgraded.scnir.json"
    write_scnir(input_path, _valid_document())

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "upgrade",
            str(input_path),
            "--output",
            str(output_path),
        ],
    ):
        rc = main()

    assert rc == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == scnir_to_dict(_valid_document())
    assert "SC-NIR upgraded" in capsys.readouterr().out
