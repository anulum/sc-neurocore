# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine retention and purge

"""Retention planning and prune-candidate purge contracts."""

from __future__ import annotations

from tests.studio_audit_quarantine_support import *  # noqa: F403

def test_build_studio_audit_quarantine_archive_retention_plan_marks_old_archives(
    tmp_path: Path,
) -> None:
    """Retention planning marks newest valid archive jobs for retention."""

    old_result = _archive_result_for_job(tmp_path, "sj_old")
    new_result = _archive_result_for_job(tmp_path, "sj_new")
    records = (
        _archive_record(
            job_id="sj_old",
            result=old_result,
            created_at_utc="2026-06-20T00:00:00Z",
            finished_at_utc="2026-06-20T00:00:01Z",
        ),
        _archive_record(
            job_id="sj_other",
            result=new_result,
            created_at_utc="2026-06-21T00:00:00Z",
            finished_at_utc="2026-06-21T00:00:01Z",
            owner="studio-evidence",
        ),
        _archive_record(
            job_id="sj_failed",
            result=None,
            created_at_utc="2026-06-21T01:00:00Z",
            finished_at_utc="2026-06-21T01:00:01Z",
            status="failed",
        ),
        _archive_record(
            job_id="sj_new",
            result=new_result,
            created_at_utc="2026-06-22T00:00:00Z",
            finished_at_utc="2026-06-22T00:00:01Z",
        ),
    )

    plan = build_studio_audit_quarantine_archive_retention_plan(
        records,
        retain_latest=1,
    ).to_public_dict()

    assert plan["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION
    assert plan["archive_count"] == 2
    assert plan["retain_count"] == 1
    assert plan["prune_candidate_count"] == 1
    assert plan["skipped_record_count"] == 1
    entries = cast(list[dict[str, object]], plan["entries"])
    assert entries[0]["job_id"] == "sj_new"
    assert entries[0]["disposition"] == "retain"
    assert entries[1]["job_id"] == "sj_old"
    assert entries[1]["disposition"] == "prune_candidate"
    assert str(tmp_path) not in json.dumps(plan)

def test_build_studio_audit_quarantine_archive_retention_plan_rejects_zero_retain(
    tmp_path: Path,
) -> None:
    """Retention planning fails closed on non-positive retain counts."""

    with pytest.raises(ValueError, match="archive_retention_retain_latest_invalid"):
        build_studio_audit_quarantine_archive_retention_plan(
            (
                _archive_record(
                    job_id="sj_archive",
                    result=_archive_result_for_job(tmp_path, "sj_archive"),
                    created_at_utc="2026-06-20T00:00:00Z",
                    finished_at_utc="2026-06-20T00:00:01Z",
                ),
            ),
            retain_latest=0,
        )

@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.__setitem__("schema_version", "unsupported"),
        lambda result: result.__setitem__("archive_id", ""),
        lambda result: result.__setitem__("summary", []),
        lambda result: cast(dict[str, object], result["summary"]).__setitem__(
            "event_count",
            "1",
        ),
        lambda result: result.pop("artifact_paths", None),
        lambda result: result.__setitem__("artifact_paths", [""]),
    ],
)
def test_build_studio_audit_quarantine_archive_retention_plan_skips_malformed_jobs(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    """Retention planning skips malformed archive job results."""

    result = _archive_result_for_job(tmp_path, "sj_malformed")
    mutation(result)

    plan = build_studio_audit_quarantine_archive_retention_plan(
        (
            _archive_record(
                job_id="sj_malformed",
                result=result,
                created_at_utc="2026-06-20T00:00:00Z",
                finished_at_utc="2026-06-20T00:00:01Z",
            ),
        ),
        retain_latest=1,
    ).to_public_dict()

    assert plan["archive_count"] == 0
    assert plan["skipped_record_count"] == 1
    assert plan["entries"] == []

def test_purge_studio_audit_quarantine_archive_prune_candidates_purges_old_jobs(
    tmp_path: Path,
) -> None:
    """Archive purge removes only retention prune candidates."""

    old_result = _archive_result_for_job(tmp_path, "sj_old")
    new_result = _archive_result_for_job(tmp_path, "sj_new")
    purged_job_ids: list[str] = []
    records = (
        _archive_record(
            job_id="sj_old",
            result=old_result,
            created_at_utc="2026-06-20T00:00:00Z",
            finished_at_utc="2026-06-20T00:00:01Z",
        ),
        _archive_record(
            job_id="sj_new",
            result=new_result,
            created_at_utc="2026-06-21T00:00:00Z",
            finished_at_utc="2026-06-21T00:00:01Z",
        ),
    )

    def purge_job(job_id: str) -> StudioJobRecord:
        purged_job_ids.append(job_id)
        return records[0]

    result = purge_studio_audit_quarantine_archive_prune_candidates(
        records,
        purge_job=purge_job,
        retain_latest=1,
    ).to_public_dict()

    assert result["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_PURGE_SCHEMA_VERSION
    assert result["purged_archive_count"] == 1
    assert result["retained_archive_count"] == 1
    assert result["skipped_record_count"] == 0
    assert purged_job_ids == ["sj_old"]
    purged_entries = cast(list[dict[str, object]], result["purged_entries"])
    retained_entries = cast(list[dict[str, object]], result["retained_entries"])
    assert purged_entries[0]["job_id"] == "sj_old"
    assert retained_entries[0]["job_id"] == "sj_new"
    assert str(tmp_path) not in json.dumps(result)
