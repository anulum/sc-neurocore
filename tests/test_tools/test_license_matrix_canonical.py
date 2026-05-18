# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - canonical model/data licence matrix test

from __future__ import annotations

import hashlib
from pathlib import Path

from tools import license_matrix


def _canonical_matrix_path() -> Path:
    return Path(__file__).resolve().parents[2] / "security" / "model_data_license_matrix.json"


def _validate_entry(row: dict[str, object]) -> None:
    source_uri = row["source_uri"]
    provenance = row["provenance"]
    assert isinstance(source_uri, str)
    assert source_uri
    assert isinstance(provenance, dict)
    provenance_source = provenance.get("source")
    assert isinstance(provenance_source, str)
    assert provenance_source.strip()
    assert provenance.get("sha256", provenance.get("hash"))
    assert isinstance(provenance.get("sha256", provenance.get("hash")), str)


def _repository_blob_path(source_uri: str) -> Path | None:
    marker = "/blob/main/"
    if marker not in source_uri:
        return None
    relative_path = source_uri.split(marker, maxsplit=1)[1]
    return Path(__file__).resolve().parents[2] / relative_path


def test_canonical_model_data_matrix_file_validates_and_reconciles_legal_policies() -> None:
    matrix_path = _canonical_matrix_path()
    assert matrix_path.exists()

    matrix = license_matrix.validate_license_matrix_file(matrix_path)

    assert matrix.project.license_identifier == "AGPL-3.0-or-later"
    assert matrix.project.commercial_license_available is True
    assert matrix.project.all_rights_reserved is True
    assert matrix.project.commercial_use == "requires_license"
    assert "all rights reserved" in matrix.project.ownership_notice.lower()

    has_pretrained_or_weights = any(
        entry.entry_type in {"pretrained_model", "model_weights"} for entry in matrix.entries
    )
    has_dataset = any(entry.entry_type == "dataset" for entry in matrix.entries)
    assert has_pretrained_or_weights
    assert has_dataset

    for entry in matrix.entries:
        assert entry.entry_id
        row = entry.as_dict()
        _validate_entry(row)
        provenance = row["provenance"]
        assert isinstance(provenance, dict)
        if provenance.get("hash_type") == "artifact-sha256":
            source_uri = row["source_uri"]
            assert isinstance(source_uri, str)
            blob_path = _repository_blob_path(source_uri)
            assert blob_path is not None
            assert blob_path.exists()
            digest = hashlib.sha256(blob_path.read_bytes()).hexdigest()
            assert digest == provenance["sha256"]
