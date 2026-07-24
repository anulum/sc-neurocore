# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine restore

"""Restore write and invalid-archive rejection contracts."""

from __future__ import annotations

from tests.studio_audit_quarantine_support import *  # noqa: F403


def test_write_studio_audit_quarantine_restore_writes_jsonl_and_manifest(
    tmp_path: Path,
) -> None:
    """Restore writer materializes validated archive rows as job artifacts."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    result = write_studio_audit_quarantine_restore(
        _archive_context(tmp_path / "restore"),
        archive_payload=archive_payload,
        manifest_payload=manifest_payload,
        clock=lambda: datetime(2026, 6, 22, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    restore_root = tmp_path / "restore" / "job" / "evidence" / "audit-quarantine"
    restore_rows = restore_root.joinpath("restore.jsonl").read_text(encoding="utf-8")
    restore_manifest = json.loads(
        restore_root.joinpath("restore-manifest.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION
    assert payload["archive_id"] == "saqa_sj_quarantine"
    assert result.artifact_paths == (
        "evidence/audit-quarantine/restore.jsonl",
        "evidence/audit-quarantine/restore-manifest.json",
    )
    assert json.loads(restore_rows)["event_hash"] == "1" * 64
    assert (
        restore_manifest["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION
    )
    assert restore_manifest["summary"]["event_count"] == 1
    assert restore_manifest["summary"]["restored_at_utc"] == "2026-06-22T00:00:00Z"
    assert str(tmp_path) not in json.dumps(payload)


def test_write_studio_audit_quarantine_restore_rejects_invalid_archive(
    tmp_path: Path,
) -> None:
    """Restore writer rejects archives that fail validation."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    manifest_payload["archive_id"] = "saqa_other"

    with pytest.raises(ValueError, match="archive_restore_validation_failed"):
        write_studio_audit_quarantine_restore(
            _archive_context(tmp_path / "restore"),
            archive_payload=archive_payload,
            manifest_payload=manifest_payload,
        )
