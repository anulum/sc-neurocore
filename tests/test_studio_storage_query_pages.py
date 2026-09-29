# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded full record query pages

"""Verify complete large pages and their cursors over real SQLite and sockets."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_query_client import (
    QueryReader,
    exchange_query,
    query_request,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_generation_support import Authority
from tests.test_studio_training_config_storage import _configuration, _create

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))


@pytest.mark.parametrize("count,limit", [(0, 1000), (3, 1), (5, 1000)])
def test_full_event_pages_preserve_creation_order_and_every_snapshot(
    tmp_path: Path, count: int, limit: int
) -> None:
    """Incremental page cutoff preserves complete records and the first excluded row."""
    config = _configuration(tmp_path / "recordings", count=1000)
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    authority = Authority(ledger, frame_max_bytes=512)
    try:
        ids = ["sj_" + f"{100 - index:016x}" for index in range(count)]
        for job_id in ids:
            _create(ledger, config, job_id=job_id)
        records = ledger.list_records()
        one = (
            len(
                json.dumps(
                    records[0].to_public_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
            )
            if records
            else 1024
        )
        budget = 2 * one + 512
        authority.services = replace(authority.services, max_view_content_bytes=budget)
        received: list[dict[str, JsonValue]] = []
        after = None
        queries = 0
        while True:
            page = exchange_query(
                authority.connect(),
                query_request("default", "records", requester=ADMIN, limit=limit, after=after),
                expected_service_uid=os.getuid(),
                max_bytes=512,
                max_content_bytes=budget,
                deadline=time.monotonic() + 10,
            )
            received.extend(page.items)
            queries += 1
            if page.next_after is None:
                break
            assert page.next_after == page.items[-1]["job_id"]
            after = page.next_after
        assert received == [record.to_public_dict() for record in records]
        assert queries == (1 if count == 0 else 3)
        reader = QueryReader(
            authority.connect,
            workspace="default",
            storage_uid=os.getuid(),
            max_bytes=512,
            max_content_bytes=budget,
            timeout_seconds=10.0,
        )
        assert reader.records(ADMIN) == records
    finally:
        authority.join()
        ledger.close()


def test_purge_page_budget_preserves_cursor_and_refuses_an_impossible_item(tmp_path: Path) -> None:
    """Actual purge rows are paginated by content; a tiny budget never drops a row."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    authority = Authority(ledger, frame_max_bytes=512)
    try:
        with ledger.transaction() as connection:
            connection.executemany(
                "INSERT INTO job_purges VALUES(?,?,?,?,?)",
                [("sj_" + f"{i:016x}", "supervisor", 7, i, "prepared") for i in range(5)],
            )
        reader = QueryReader(
            authority.connect,
            workspace="default",
            storage_uid=os.getuid(),
            max_bytes=512,
            timeout_seconds=10.0,
        )
        authority.services = replace(authority.services, max_view_content_bytes=512)
        ids: list[str] = []
        after = None
        while True:
            page = reader.purges(ADMIN, limit=1000, after=after)
            ids.extend(item.job_id for item in page.purges)
            if page.next_after is None:
                break
            after = page.next_after
        assert ids == ["sj_" + f"{i:016x}" for i in range(5)]
        authority.services = replace(authority.services, max_view_content_bytes=200)
        with pytest.raises(EOFError):
            reader.purges(ADMIN, limit=1000, after=None)
        authority.join(expected=(ValueError,))
    finally:
        ledger.close()
