# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public identity conversion over retained native evidence

"""Keep real proof metadata intact while rejecting unsafe identity conversions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_public_identity import validate_identity_conversion

ROOT = Path(__file__).resolve().parents[2]
INDEX = "benchmarks/results/formal/index.json"
MANIFEST = "benchmarks/results/formal/public_identity_conversion.json"
AMARI = "hdl/formal/catalogue/sc_amari_field.sby"
CHAOTIC = "hdl/formal/catalogue/sc_chaotic_map.sby"
CHIALVO = "hdl/formal/catalogue/sc_chialvo_map.sby"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _value_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Copy the nine real converted measurements and their explicit public bindings."""
    manifest = read_object(ROOT / MANIFEST)
    names = {MANIFEST}
    for raw in cast(list[object], manifest["records"]):
        names.add(str(object_fields(raw)["path"]))
    index = read_object(ROOT / INDEX)
    jobs = object_fields(index["jobs"])
    for job in [AMARI, CHAOTIC, CHIALVO]:
        entry = object_fields(jobs[job])
        for field in ["review", "nonvacuity_review"]:
            names.add(str(entry[field]))
        for raw in cast(list[object], entry.get("predecessors", [])):
            names.add(str(object_fields(raw)["review"]))
        if "solver_error" in entry:
            names.add(str(object_fields(entry["solver_error"])["review"]))
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path, jobs, manifest


def _bind(repository: Path, entry: dict[str, object], manifest: dict[str, object]) -> None:
    _write(repository / MANIFEST, manifest)
    entry["identity_conversion"] = {"path": MANIFEST, "sha256": digest(repository / MANIFEST)}


@pytest.mark.parametrize("job", [AMARI, CHAOTIC, CHIALVO])
def test_real_measurement_conversion_keeps_native_identities(
    case: tuple[Path, dict[str, object], dict[str, object]], job: str
) -> None:
    """Validate identity conversion without claiming a new proof or review verdict."""
    repository, jobs, _manifest = case
    entry = object_fields(jobs[job])
    before = (repository / str(entry["receipt"])).read_bytes()

    validate_identity_conversion(repository, entry)

    assert (repository / str(entry["receipt"])).read_bytes() == before
    primary = read_object(repository / str(entry["receipt"]))
    audit = read_object(repository / str(entry["nonvacuity"]))
    assert (
        object_fields(primary["measurement"])["invocation_id"]
        != object_fields(audit["measurement"])["invocation_id"]
    )
    assert read_object(repository / str(entry["review"]))["verdict"] in {"OPEN", "CLEAR"}


def test_unconverted_descriptive_metadata_needs_no_conversion_alias(
    case: tuple[Path, dict[str, object], dict[str, object]],
) -> None:
    """Accept identity metadata that has neither a replaced unit nor a private lease."""
    repository, jobs, _manifest = case
    entry = {"receipt": object_fields(jobs[AMARI])["receipt"]}
    validate_identity_conversion(repository, entry)


@pytest.mark.parametrize("field", ["receipt", "nonvacuity"])
def test_invalid_public_reference_refuses(
    case: tuple[Path, dict[str, object], dict[str, object]], field: str
) -> None:
    """Refuse absent artifact locations through the production metadata API."""
    repository, _jobs, _manifest = case
    with pytest.raises(ValueError, match="Invalid public evidence reference"):
        validate_identity_conversion(repository, {field: None})


@pytest.mark.parametrize("mutation", ["unit", "nonce", "missing_manifest", "manifest_digest"])
def test_coordination_identity_and_unbound_conversion_refuse(
    case: tuple[Path, dict[str, object], dict[str, object]], mutation: str
) -> None:
    """Reject coordination leakage and unbound labels before downstream proof checks."""
    repository, jobs, _manifest = case
    entry = object_fields(jobs[CHAOTIC])
    receipt_path = repository / str(entry["receipt"])
    receipt = read_object(receipt_path)
    if mutation == "unit":
        measurement = object_fields(receipt["measurement"])
        # An obsolete seat-bearing service name is a negative input only.
        measurement["unit"] = "proof-012345ab-20260101T000000.service"
        receipt["measurement"] = measurement
        _write(receipt_path, receipt)
    elif mutation == "nonce":
        custody = object_fields(receipt["custody"])
        custody["released_nonce"] = "obsolete-private-lease"
        receipt["custody"] = custody
        _write(receipt_path, receipt)
    elif mutation == "missing_manifest":
        entry.pop("identity_conversion")
    else:
        entry["identity_conversion"] = {"path": MANIFEST, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_identity_conversion(repository, entry)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "records",
        "public_digest",
        "original_digest",
        "same_digest",
        "kind",
        "changes",
        "field",
        "repeated_field",
        "original_field_digest",
        "remove_operation",
        "remove_public_digest",
        "lease_binding",
        "replace_operation",
        "public_field_digest",
        "nonce_removal_missing",
        "unit_conversion_missing",
        "unit_label",
        "producer",
        "preserved_fact",
        "absent_field",
        "duplicate_record",
        "omitted_record",
        "review_binding",
    ],
)
def test_real_conversion_refuses_isolated_binding_and_scope_changes(
    case: tuple[Path, dict[str, object], dict[str, object]], mutation: str
) -> None:
    """Rebind outer hashes while isolating malformed conversion or changed native facts."""
    repository, jobs, manifest = case
    entry = object_fields(jobs[CHAOTIC])
    records = cast(list[dict[str, object]], manifest["records"])
    record = next(row for row in records if row["path"] == entry["receipt"])
    audit_record = next(row for row in records if row["path"] == entry["nonvacuity"])
    target = repository / str(record["path"])
    payload = read_object(target)
    changes = cast(list[dict[str, object]], record["changes"])
    if mutation == "schema":
        manifest["schema_version"] = "obsolete"
    elif mutation == "records":
        manifest["records"] = []
    elif mutation == "public_digest":
        record["public_artifact_sha256"] = "0" * 64
    elif mutation == "original_digest":
        record["original_artifact_sha256"] = "invalid"
    elif mutation == "same_digest":
        record["original_artifact_sha256"] = record["public_artifact_sha256"]
    elif mutation == "kind":
        record["kind"] = {}
    elif mutation == "changes":
        record["changes"] = []
    elif mutation == "field":
        changes[0]["field"] = "measurement.wall_seconds"
    elif mutation == "repeated_field":
        changes.append(changes[0].copy())
    elif mutation == "original_field_digest":
        changes[0]["original_value_sha256"] = "invalid"
    elif mutation == "remove_operation":
        changes[-1]["operation"] = "replace"
    elif mutation == "remove_public_digest":
        changes[-1]["public_value_sha256"] = "0" * 64
    elif mutation == "lease_binding":
        custody = object_fields(payload["custody"])
        custody["lease_release_sha256"] = "invalid"
        payload["custody"] = custody
    elif mutation == "replace_operation":
        changes[0]["operation"] = "remove"
    elif mutation == "public_field_digest":
        changes[0]["public_value_sha256"] = "0" * 64
    elif mutation == "nonce_removal_missing":
        record["changes"] = changes[:-1]
    elif mutation == "unit_conversion_missing":
        record["changes"] = changes[1:]
    elif mutation == "unit_label":
        measurement = object_fields(payload["measurement"])
        measurement["unit"] = "<FORMAL_AUDIT_UNIT>"
        payload["measurement"] = measurement
        changes[0]["public_value_sha256"] = _value_hash(measurement["unit"])
    elif mutation in {"producer", "absent_field"}:
        record = audit_record
        target = repository / str(record["path"])
        payload = read_object(target)
        if mutation == "producer":
            payload["producer"] = "undocumented producer"
            for change in cast(list[dict[str, object]], record["changes"]):
                if change["field"] == "producer":
                    change["public_value_sha256"] = _value_hash(payload["producer"])
        else:
            payload.pop("producer")
    elif mutation == "preserved_fact":
        measurement = object_fields(payload["measurement"])
        measurement["peak_memory_bytes"] = 1
        payload["measurement"] = measurement
    elif mutation == "duplicate_record":
        records.append(record.copy())
    elif mutation == "omitted_record":
        records.remove(record)
    else:
        review_path = repository / str(entry["review"])
        review = read_object(review_path)
        review["identity_conversion_sha256"] = "0" * 64
        _write(review_path, review)
    if mutation in {"lease_binding", "unit_label", "producer", "preserved_fact", "absent_field"}:
        _write(target, payload)
        record["public_artifact_sha256"] = digest(target)
    _bind(repository, entry, manifest)
    with pytest.raises(ValueError):
        validate_identity_conversion(repository, entry)
