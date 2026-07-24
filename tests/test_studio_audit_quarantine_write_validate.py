# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine write and validate

"""Archive write, validate acceptance, and validation defect contracts."""

from __future__ import annotations

from tests.studio_audit_quarantine_support import *  # noqa: F403


def test_write_studio_audit_quarantine_archive_writes_manifest_and_payload(
    tmp_path: Path,
) -> None:
    """Quarantine archive writer emits confined, path-free archive artifacts."""

    result = write_studio_audit_quarantine_archive(
        _archive_context(tmp_path),
        quarantine_export=_quarantine_export_payload(),
        clock=lambda: datetime(2026, 6, 21, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    archive_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "archive.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION
    assert payload["archive_id"] == "saqa_sj_quarantine"
    assert result.artifact_paths == (
        "evidence/audit-quarantine/archive.json",
        "evidence/audit-quarantine/manifest.json",
    )
    assert archive_payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION
    assert archive_payload["quarantine_export"]["event_count"] == 1
    assert archive_payload["summary"]["reason_counts"] == {"legacy_or_unverifiable_rows": 1}
    assert manifest_payload["summary"] == archive_payload["summary"]
    assert str(tmp_path) not in json.dumps(payload)


def test_validate_studio_audit_quarantine_archive_accepts_writer_output(
    tmp_path: Path,
) -> None:
    """Archive validation accepts the writer's archive and manifest pair."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION
    assert validation["valid"] is True
    assert validation["archive_id"] == "saqa_sj_quarantine"
    assert validation["errors"] == []
    validation_summary = cast(dict[str, object], validation["summary"])
    assert validation_summary["event_count"] == 1
    assert validation_summary["reason_counts"] == {"legacy_or_unverifiable_rows": 1}
    assert str(tmp_path) not in json.dumps(validation)


def test_validate_studio_audit_quarantine_archive_reports_manifest_mismatch(
    tmp_path: Path,
) -> None:
    """Archive validation reports mismatched companion manifests."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    manifest_payload["archive_id"] = "saqa_other"
    manifest_payload["summary"] = {}

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["archive_id"] == "saqa_sj_quarantine"
    assert validation["errors"] == [
        "manifest_archive_id_mismatch",
        "manifest_summary_mismatch",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_errors"),
    [
        (
            lambda payload: payload.__setitem__(
                "schema_version",
                "studio.audit-quarantine-archive.v0",
            ),
            ["manifest_schema_unsupported"],
        ),
        (
            lambda payload: payload.__setitem__("artifact_count", 2),
            ["manifest_artifact_count_invalid"],
        ),
        (
            lambda payload: payload.__setitem__("entries", {}),
            ["manifest_archive_entry_missing"],
        ),
        (
            lambda payload: payload.__setitem__("entries", ["invalid"]),
            ["manifest_archive_entry_missing"],
        ),
        (
            lambda payload: payload.__setitem__(
                "entries",
                [{"type": "other", "bundle_path": "evidence/other.json"}],
            ),
            ["manifest_archive_entry_missing"],
        ),
    ],
)
def test_validate_studio_audit_quarantine_archive_reports_manifest_defects(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    expected_errors: list[str],
) -> None:
    """Archive validation reports stable manifest defect codes."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    mutation(manifest_payload)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == expected_errors


@pytest.mark.parametrize(
    "manifest_payload",
    [
        {"bad": float("nan")},
        cast(dict[str, object], {1: "non-string-key"}),
        {"bad": object()},
    ],
)
def test_validate_studio_audit_quarantine_archive_rejects_non_json_manifest(
    tmp_path: Path,
    manifest_payload: dict[str, object],
) -> None:
    """Archive validation rejects manifests that cannot be JSON payloads."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == ["manifest_not_json"]


def test_validate_studio_audit_quarantine_archive_reports_summary_mismatch(
    tmp_path: Path,
) -> None:
    """Archive validation recomputes summary fields before import."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)
    archive_summary = cast(dict[str, object], archive_payload["summary"])
    archive_summary["event_count"] = 999

    validation = validate_studio_audit_quarantine_archive(archive_payload).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == ["archive_summary_mismatch"]
    validation_summary = cast(dict[str, object], validation["summary"])
    assert validation_summary["event_count"] == 1


def test_write_studio_audit_quarantine_archive_rejects_malformed_export(
    tmp_path: Path,
) -> None:
    """Quarantine archive writer rejects unsupported export schemas."""

    with pytest.raises(ValueError, match="export_schema_unsupported"):
        write_studio_audit_quarantine_archive(
            _archive_context(tmp_path),
            quarantine_export=_quarantine_export_payload()
            | {"schema_version": "studio.audit.export.v1"},
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda payload: payload.__setitem__("events", {}), "export_events_invalid"),
        (lambda payload: payload.__setitem__("event_count", 2), "export_event_count"),
        (lambda payload: payload.__setitem__("retained_event_count", 0), "retained"),
        (lambda payload: payload.__setitem__("truncated", "false"), "truncated"),
        (lambda payload: payload.__setitem__("quarantine_reason", 7), "reason"),
        (lambda payload: payload.__setitem__("events", ["invalid"]), "export_event"),
        (
            lambda payload: payload.__setitem__("events", [{"action": "studio.test"}]),
            "export_event",
        ),
    ],
)
def test_write_studio_audit_quarantine_archive_rejects_invalid_export_shapes(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    error_match: str,
) -> None:
    """Quarantine archive writer validates each public export field."""

    export_payload = _quarantine_export_payload()
    mutation(export_payload)

    with pytest.raises(ValueError, match=error_match):
        write_studio_audit_quarantine_archive(
            _archive_context(tmp_path),
            quarantine_export=export_payload,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda payload: payload.__setitem__(
                "schema_version",
                "studio.audit-quarantine-archive.v0",
            ),
            "archive_schema_unsupported",
        ),
        (lambda payload: payload.__setitem__("archive_id", ""), "archive_id_invalid"),
        (
            lambda payload: payload.__setitem__("archived_at_utc", "2026-06-21"),
            "archive_timestamp_invalid",
        ),
        (
            lambda payload: payload.__setitem__("quarantine_export", {}),
            "export_schema_unsupported",
        ),
        (
            lambda payload: payload.__setitem__("quarantine_export", []),
            "archive_export_missing",
        ),
        (lambda payload: payload.__setitem__("summary", []), "archive_summary_missing"),
    ],
)
def test_validate_studio_audit_quarantine_archive_rejects_invalid_archive_shapes(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    expected_error: str,
) -> None:
    """Archive validation returns stable error codes for invalid archives."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)
    mutation(archive_payload)

    validation = validate_studio_audit_quarantine_archive(archive_payload).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == [expected_error]
    assert validation["summary"] is None
