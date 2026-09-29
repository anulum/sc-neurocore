# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded storage query page materialisation

"""Read complete pages incrementally without loading the entire result set."""

from __future__ import annotations

import json
from pydantic import JsonValue

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import record_from_row
from sc_neurocore.studio.platform.storage_query_protocol import (
    QUERY_SCHEMA_VERSION,
    StorageQueryRequest,
    StorageQueryResponse,
    encode_query_message,
)


def _page_response(
    request: StorageQueryRequest,
    items: list[dict[str, JsonValue]],
    cursor: str | None,
) -> StorageQueryResponse:
    """Build one complete answered page under the original inner grammar."""
    return StorageQueryResponse(
        schema_version=QUERY_SCHEMA_VERSION,
        operation="query",
        request_id=request.request_id,
        view=request.view,
        status="ok",
        items=tuple(items),
        summary=None,
        next_after=cursor,
    )


def fit_page(
    request: StorageQueryRequest,
    items: list[dict[str, JsonValue]],
    more: bool,
    *,
    max_bytes: int,
) -> bytes:
    """Encode the longest prefix of ``items`` whose response fits the content limit.

    Each item is measured by its own compact encoding plus one separator, and
    the envelope is measured with a cursor of full length, so the estimate
    never undercounts the encoded page. A shortened page carries a cursor to
    its last item, so the API continues where it stopped.

    Raises
    ------
    ValueError
        A single item does not fit the content limit: a configuration fault.
    """
    placeholder = "sj_" + "0" * 16
    budget = max_bytes - len(encode_query_message(_page_response(request, [], placeholder)))
    count = 0
    for item in items:
        size = len(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)) + 1
        if size > budget:
            break
        budget -= size
        count += 1
    if items and count == 0:
        raise ValueError("one storage query item exceeds the content limit")
    page = items[:count]
    cursor = str(page[-1]["job_id"]) if page and (more or count < len(items)) else None
    return encode_query_message(_page_response(request, page, cursor))


def read_record_page(
    ledger: StudioJobLedger,
    request: StorageQueryRequest,
    *,
    max_bytes: int,
) -> tuple[list[dict[str, JsonValue]], bool] | None:
    """Materialize only records that fit the independently bounded page.

    Parameters
    ----------
    ledger : StudioJobLedger
        Authorized service ledger, read without transitions or recovery.
    request : StorageQueryRequest
        Workspace-bound records query and creation-order cursor.
    max_bytes : int
        Trusted total page ceiling, distinct from the wire frame limit.

    Returns
    -------
    tuple or None
        Complete records and whether another record remains; None for an unknown
        cursor. A cursor always names the last returned record, so budget cutoff
        never loses the first excluded item.

    Raises
    ------
    ValueError
        One complete record cannot fit the configured total page budget.

    Notes
    -----
    At most one next row is decoded beyond the returned page's memory. SQLite
    rows are consumed incrementally; a thousand 64 MiB declarations are never
    fetched into one Python list before the page budget is applied.
    """
    budget = max_bytes - len(encode_query_message(_page_response(request, [], "sj_" + "0" * 16)))
    items: list[dict[str, JsonValue]] = []
    with ledger.transaction() as connection:
        cursor: tuple[str, int] = ("", 0)
        if request.after is not None:
            row = connection.execute(
                "SELECT created_at_utc, rowid FROM jobs WHERE job_id = ? AND workspace = ?",
                (request.after, request.workspace),
            ).fetchone()
            if row is None:
                return None
            cursor = (str(row[0]), int(row[1]))
        rows = connection.execute(
            "SELECT * FROM jobs WHERE workspace = ? AND (created_at_utc, rowid) > (?, ?) "
            "ORDER BY created_at_utc, rowid LIMIT ?",
            (request.workspace, *cursor, request.limit + 1),
        )
        try:
            for row in rows:
                if len(items) == request.limit:
                    return items, True
                decoded = json.loads(
                    json.dumps(record_from_row(row).to_public_dict(), allow_nan=False)
                )
                item = {str(key): value for key, value in decoded.items()}
                size = (
                    len(
                        json.dumps(
                            item, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode("utf-8")
                    )
                    + 1
                )
                if size > budget:
                    if not items:
                        raise ValueError("one storage query item exceeds the content limit")
                    return items, True
                budget -= size
                items.append(item)
        finally:
            rows.close()
    return items, False
