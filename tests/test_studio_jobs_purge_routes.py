# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Operator purge inspection contracts

"""Exercise the real paginated operator route without granting purge mutation authority."""

import os
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import StudioJobManager

ADMIN = {"x-studio-principal": "operator", "x-studio-roles": "studio.admin"}


def test_admin_pages_actual_purge_journal_without_recovery(tmp_path: Path) -> None:
    """All phases survive reads, with bounded lexical pages and no private identity leakage."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True, job_root_path=str(tmp_path / "jobs")
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    phases = ("prepared", "committed", "cleanup_started", "removed", "ambiguous")
    with manager._ledger.transaction() as connection:
        connection.executemany(
            "INSERT INTO job_purges VALUES(?,?,?,?,?)",
            [
                (
                    f"sj_{index:016x}",
                    "private-supervisor",
                    42 if index else None,
                    index if index else None,
                    phase,
                )
                for index, phase in enumerate(phases)
            ],
        )
    connection = manager._ledger.connection()
    before = list(connection.iterdump())
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/studio/jobs/purges").status_code == 401
        assert (
            client.get(
                "/api/studio/jobs/purges", headers={"x-studio-principal": "reader"}
            ).status_code
            == 403
        )
        response = client.get("/api/studio/jobs/purges?limit=2", headers=ADMIN)
        assert response.status_code == 200
        assert response.json() == {
            "schema_version": "studio.jobs.purges.v1",
            "purges": [
                {
                    "job_id": f"sj_{i:016x}",
                    "phase": phases[i],
                    "device": 42 if i else None,
                    "inode": i if i else None,
                }
                for i in range(2)
            ],
            "next_after": "sj_0000000000000001",
        }
        second = client.get(
            "/api/studio/jobs/purges",
            headers=ADMIN,
            params={"limit": 2, "after": response.json()["next_after"]},
        )
        assert [r["phase"] for r in second.json()["purges"]] == list(phases[2:4])
        third = client.get(
            "/api/studio/jobs/purges",
            headers=ADMIN,
            params={"limit": 2, "after": second.json()["next_after"]},
        )
        assert [r["phase"] for r in third.json()["purges"]] == ["ambiguous"]
        assert third.json()["next_after"] is None
        empty = client.get("/api/studio/jobs/purges?after=sj_ffffffffffffffff", headers=ADMIN)
        assert empty.json()["purges"] == [] and empty.json()["next_after"] is None
        for page in (response, second, third, empty):
            assert "private-supervisor" not in page.text
            assert str(tmp_path) not in page.text
        assert client.get("/api/studio/jobs/status").json()["pending_purge_count"] == 5
    assert list(connection.iterdump()) == before


@pytest.mark.parametrize("query", ["limit=0", "limit=1001", "limit=no", "after=../private"])
def test_purge_page_rejects_invalid_http_bounds(tmp_path: Path, query: str) -> None:
    """Reject invalid operator queries through the real router, without leaking paths."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True, job_root_path=str(tmp_path)
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(f"/api/studio/jobs/purges?{query}", headers=ADMIN)
    assert response.status_code == 422
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("limit", [0, 1001, True])
def test_manager_purge_page_enforces_its_own_limit(tmp_path: Path, limit: int) -> None:
    """Direct manager users cannot bypass the SQL page bound enforced at HTTP."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    with pytest.raises(ValueError, match="page limit"):
        manager.purge_snapshot(limit=limit)
    with pytest.raises(ValueError, match="cursor"):
        manager.purge_snapshot(after="not-a-job")


@pytest.mark.skipif(os.geteuid() == 0, reason="file modes do not bind the root identity")
def test_purge_inspection_refuses_failed_audit_before_read(tmp_path: Path) -> None:
    """A failed existing authorization audit must not reach the journal reader.

    The audit log file is really unwritable. The journal holds a row that the
    reader would reject as an invalid page (422), so only an audit refusal
    ahead of any journal read answers 503.
    """
    audit = tmp_path / "audit" / "audit.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            audit_log_path=str(audit),
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES('sj_0000000000000001','owner','not-a-device',1,"
            "'prepared')"
        )
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.touch()
    audit.chmod(0o400)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        try:
            response = client.get("/api/studio/jobs/purges", headers=ADMIN)
        finally:
            audit.chmod(0o600)
        # With a writable audit log the same request reads the damaged journal.
        control = client.get("/api/studio/jobs/purges", headers=ADMIN)
    assert response.status_code == 503
    assert response.json() == {"detail": "audit_append_failed"}
    assert control.status_code == 422
