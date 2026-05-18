# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for license data matrix validator

from __future__ import annotations

import json
from typing import Any, cast
from pathlib import Path

import pytest

from tools import license_matrix


def _valid_project_block() -> dict[str, object]:
    return {
        "license_identifier": "AGPL-3.0-or-later",
        "commercial_license_available": True,
        "ownership_notice": "Copyright (c) Concepts 2026. All rights reserved.",
        "all_rights_reserved": True,
        "commercial_use": "requires_license",
    }


def _valid_entries() -> list[dict[str, object]]:
    return [
        {
            "entry_id": "pretrained/mnist.pt",
            "entry_type": "pretrained_model",
            "license_identifier": "CC-BY-4.0",
            "source_uri": "https://example.com/pretrained/mnist.pt",
            "provenance": {
                "source": "model-zoo",
                "version": "v1.0",
                "sha256": "a" * 64,
            },
            "redistribution_status": "restricted",
            "attribution_requirements": [
                "cite paper: NeuronFlow et al., 2026",
                "retain project ownership notice",
            ],
            "commercial_use": "requires_license",
            "commercial_license_required": True,
        },
        {
            "entry_id": "shd-dataset",
            "entry_type": "dataset",
            "license_identifier": "CC0-1.0",
            "source_uri": "https://example.com/datasets/shd.zip",
            "provenance": {
                "source": "dataset-archive",
                "doi": "10.1234/shd",
                "sha256": "b" * 64,
            },
            "attribution_requirements": [
                "cite dataset DOI",
            ],
            "commercial_use": "allowed",
            "commercial_license_required": False,
        },
        {
            "entry_id": "model/weights-int8.json",
            "entry_type": "model_weights",
            "license_identifier": "CC-BY-4.0",
            "source_uri": "https://example.com/model/weights.json",
            "provenance": {
                "source": "internal-pipeline",
                "commit": "abc123",
                "sha256": "c" * 64,
            },
            "redistribution_status": "allowed",
            "attribution_requirements": [
                "keep pretrained attribution footer",
            ],
            "commercial_use": "requires_license",
            "commercial_license_required": True,
        },
    ]


def _write_matrix(tmp_path: Path, payload: dict[str, object]) -> Path:
    matrix_path = tmp_path / "license_matrix.json"
    matrix_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return matrix_path


def _base_matrix() -> dict[str, Any]:
    return {
        "project": _valid_project_block(),
        "entries": _valid_entries(),
    }


def test_validate_matrix_rejects_missing_licence_identifier(tmp_path: Path) -> None:
    payload = _base_matrix()
    payload["entries"][0]["license_identifier"] = ""  # type: ignore[index]

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(license_matrix.LicenseMatrixValidationError, match="license_identifier"):
        license_matrix.validate_license_matrix_file(path)


def test_validate_matrix_rejects_missing_artifact_source_and_provenance(tmp_path: Path) -> None:
    payload = _base_matrix()
    entries = cast(list[dict[str, Any]], payload["entries"])
    del entries[1]["source_uri"]
    del entries[1]["provenance"]

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(license_matrix.LicenseMatrixValidationError, match="source_uri|provenance"):
        license_matrix.validate_license_matrix_file(path)


def test_validate_matrix_rejects_weak_provenance_without_source_or_digest(tmp_path: Path) -> None:
    payload = _base_matrix()
    entries = cast(list[dict[str, Any]], payload["entries"])
    entries[0]["provenance"] = {"version": "v1.0"}

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(
        license_matrix.LicenseMatrixValidationError,
        match="provenance.source|sha256|hash",
    ):
        license_matrix.validate_license_matrix_file(path)


def test_validate_matrix_rejects_pretrained_without_redistribution_status(tmp_path: Path) -> None:
    payload = _base_matrix()
    del payload["entries"][0]["redistribution_status"]  # type: ignore[index]

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(
        license_matrix.LicenseMatrixValidationError,
        match="pretrained_model requires redistribution_status",
    ):
        license_matrix.validate_license_matrix_file(path)

    payload = _base_matrix()
    del payload["entries"][2]["redistribution_status"]  # type: ignore[index]

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(
        license_matrix.LicenseMatrixValidationError,
        match="model_weights requires redistribution_status",
    ):
        license_matrix.validate_license_matrix_file(path)


def test_validate_matrix_rejects_contradictory_commercial_fields(tmp_path: Path) -> None:
    payload = _base_matrix()
    project = cast(dict[str, Any], payload["project"])
    entries = cast(list[dict[str, Any]], payload["entries"])
    project["commercial_license_available"] = False
    entries[0]["commercial_use"] = "requires_license"

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(
        license_matrix.LicenseMatrixValidationError,
        match="disallows commercial licensing|commercial_license_available",
    ):
        license_matrix.validate_license_matrix_file(path)

    payload = _base_matrix()
    entries = cast(list[dict[str, Any]], payload["entries"])
    entries[1]["commercial_use"] = "requires_license"
    entries[1]["commercial_license_required"] = False

    path = _write_matrix(tmp_path, payload)

    with pytest.raises(
        license_matrix.LicenseMatrixValidationError,
        match="commercial_license_required|requires commercial licensing",
    ):
        license_matrix.validate_license_matrix_file(path)


def test_build_report_is_deterministic_and_cli_validates(tmp_path: Path) -> None:
    payload = _base_matrix()
    path = _write_matrix(tmp_path, payload)

    first = license_matrix.validate_license_matrix_file(path)
    second = license_matrix.validate_license_matrix_file(path)
    first_report = first.as_report()
    second_report = second.as_report()

    assert first_report == second_report
    assert first_report["schema_version"] == license_matrix.LICENSE_MATRIX_SCHEMA_VERSION
    assert first_report["status"] == "valid"
    assert first_report["entry_count"] == 3
    assert first_report["entries"] == json.loads(
        json.dumps(first_report["entries"], sort_keys=True)
    )


def test_cli_rejects_invalid_matrix_and_writes_report_for_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _base_matrix()
    invalid_path = _write_matrix(tmp_path, payload)
    output_path = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv", ["license-matrix", str(invalid_path), "--output-json", str(output_path)]
    )
    assert license_matrix.main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "valid"
    assert report["project"]["license_identifier"] == "AGPL-3.0-or-later"

    payload["entries"][0]["commercial_license_required"] = True
    payload["entries"][0]["commercial_use"] = "prohibited"
    invalid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["license-matrix", str(invalid_path)])

    assert license_matrix.main() == 1
