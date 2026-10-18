# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded benchmark contribution persistence

"""Apply byte, record and page budgets while serialising databank access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_neurocore.studio.workspace_lock import workspace_lock

DEFAULT_DATABANK_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_DATABANK_MAX_RECORDS = 10_000
MAX_DATABANK_PAGE_SIZE = 1000
MAX_DATABANK_RECORD_BYTES = 1024 * 1024


class BenchmarkDatabankCapacityError(ValueError):
    """A declared storage or record budget refused an operation without writing."""

    def __init__(self, *, record_too_large: bool = False) -> None:
        self.status_code = 413 if record_too_large else 507
        super().__init__(
            "benchmark_record_too_large" if record_too_large else "benchmark_databank_full"
        )


def _read_records(path: Path, *, max_bytes: int, max_records: int) -> list[bytes]:
    """Read JSONL bytes within both budgets before parsing or admitting an append."""
    if max_bytes <= 0 or max_records <= 0:
        raise ValueError("Benchmark databank budgets must be positive.")
    if not path.exists():
        return []
    rows: list[bytes] = []
    total = 0
    with path.open("rb") as stream:
        if stream.seek(0, 2) > max_bytes:
            raise BenchmarkDatabankCapacityError()
        stream.seek(0)
        while line := stream.readline(min(MAX_DATABANK_RECORD_BYTES, max_bytes) + 1):
            total += len(line)
            if total > max_bytes or len(line) > MAX_DATABANK_RECORD_BYTES:
                raise BenchmarkDatabankCapacityError()
            if line.strip():
                if len(rows) >= max_records:
                    raise BenchmarkDatabankCapacityError()
                rows.append(line)
    return rows


def append_databank(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int = DEFAULT_DATABANK_MAX_BYTES,
    max_records: int = DEFAULT_DATABANK_MAX_RECORDS,
) -> None:
    """Append one complete record only if it fits both store budgets.

    A process-shared lock covers the capacity check and append. A capacity
    refusal leaves existing bytes unchanged. Each JSON record is also limited
    to one MiB, independent of the total store budget.
    """
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    if len(encoded) > min(max_bytes, MAX_DATABANK_RECORD_BYTES):
        raise BenchmarkDatabankCapacityError(record_too_large=True)
    canonical = path.expanduser().resolve()
    with workspace_lock(canonical.parent, f"benchmark-{canonical.name}"):
        rows = _read_records(canonical, max_bytes=max_bytes, max_records=max_records)
        size = canonical.stat().st_size if canonical.exists() else 0
        if len(rows) >= max_records or size + len(encoded) > max_bytes:
            raise BenchmarkDatabankCapacityError()
        with canonical.open("ab") as stream:
            stream.write(encoded)


def read_databank(
    path: Path,
    *,
    limit: int = 100,
    offset: int = 0,
    max_bytes: int = DEFAULT_DATABANK_MAX_BYTES,
    max_records: int = DEFAULT_DATABANK_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Read one finite page while refusing a store beyond its declared budgets."""
    if not 1 <= limit <= MAX_DATABANK_PAGE_SIZE or not 0 <= offset <= max_records:
        raise ValueError("Benchmark databank page is outside the supported bounds.")
    canonical = path.expanduser().resolve()
    if not canonical.exists():
        return []
    with workspace_lock(canonical.parent, f"benchmark-{canonical.name}"):
        rows = _read_records(canonical, max_bytes=max_bytes, max_records=max_records)
        page = []
        for line in rows[offset : offset + limit]:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Benchmark databank record must be an object.")
            page.append(record)
        return page
