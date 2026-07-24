# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cli) from former test_scnir_compatibility.py

from __future__ import annotations

from tests.scnir_compatibility_support import *  # noqa: F403

def test_scnir_compatibility_cli_validates_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(REPO_ROOT)],
    )

    assert main() == 0
    assert "SC-NIR compatibility matrix valid" in capsys.readouterr().out

def test_scnir_compatibility_cli_writes_matrix_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "scnir_compatibility.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(REPO_ROOT), "--output", str(output)],
    )

    assert main() == 0
    assert f"report written: {output}" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload == json.loads(json.dumps(scnir_compatibility_matrix_dicts(), sort_keys=True))
    assert payload[0]["nir_primitive"] == "Input"

def test_scnir_closure_audit_cli_writes_versioned_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "scnir_closure_audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "closure-audit", str(REPO_ROOT), "--output", str(output)],
    )

    assert main() == 0
    assert f"report written: {output}" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.scnir.compatibility-audit.v0.2"
    assert payload["status"] == "valid"
    assert payload["primitive_count"] == len(scnir_compatibility_matrix())
    assert payload["closure_status"] == "closed_for_local_handoff"
    assert payload["closure_blocker_count"] == 0
    assert payload["parser_only_primitives"] == []
    assert payload["metadata_only_primitives"] == []
    assert payload["boundary_primitives"] == ["Input", "Output"]
    assert payload["requires_external_hardware_evidence"] is True
    assert payload["external_hardware_evidence_status"] == "not_claimed"
    assert payload["audit_evidence_file_count"] >= 1
    assert payload["matrix_sha256"]
    assert payload["audit_evidence_files"][0]["sha256"]
    assert payload["matrix"][0]["nir_primitive"] == "Input"

def test_scnir_compatibility_cli_rejects_missing_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(tmp_path)],
    )

    assert main() == 1
    assert "SC-NIR compatibility matrix invalid" in capsys.readouterr().out
