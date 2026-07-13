# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit integrity helpers

"""Integrity, quarantine, and export helpers for the Studio JSONL audit sink."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sc_neurocore.studio.platform.policy_models import (
    _AuditIntegrityReport,
    AuditSinkError,
)


class _JsonlAuditIntegrityMixin:
    """Internal integrity operations shared by the concrete JSONL audit sink."""

    _path: Path
    _last_error: str | None

    def _export_paths(self) -> tuple[Path, ...]:
        """Return retained audit paths from oldest to newest."""

        raise NotImplementedError  # pragma: no cover - subclass contract

    def _raw_export_rows(self) -> list[dict[object, object]]:
        """Load retained JSON objects before scalar public-value validation."""

        rows: list[dict[object, object]] = []
        try:
            for path in self._export_paths():
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        self._last_error = "AuditExportInvalidRow"
                        raise AuditSinkError("Studio audit export failed.")
                    rows.append(parsed)
        except json.JSONDecodeError as exc:
            self._last_error = "AuditExportInvalidJson"
            raise AuditSinkError("Studio audit export failed.") from exc
        except OSError as exc:
            self._last_error = type(exc).__name__
            raise AuditSinkError("Studio audit export failed.") from exc
        return rows

    def _public_export_row(self, parsed: dict[object, object]) -> dict[str, str | None]:
        """Validate and narrow one retained row to its public scalar shape."""

        row: dict[str, str | None] = {}
        for key, value in parsed.items():
            if not isinstance(key, str) or not (isinstance(value, str) or value is None):
                self._last_error = "AuditExportInvalidRow"
                raise AuditSinkError("Studio audit export failed.")
            row[key] = value
        return row

    def _verify_integrity(self) -> _AuditIntegrityReport:
        """Verify row hashes and chain continuity across retained logs."""

        try:
            rows = [self._public_export_row(row) for row in self._raw_export_rows()]
        except AuditSinkError:
            return _AuditIntegrityReport(False, self._last_error, None)
        previous_hash: str | None = None
        latest_hash: str | None = None
        quarantined_event_count = 0
        quarantine_reason: str | None = None
        for index, row in enumerate(rows):
            event_hash = row.get("event_hash")
            if event_hash is None:
                quarantined_event_count += 1
                quarantine_reason = "legacy_or_unverifiable_rows"
                previous_hash = None
                continue
            expected_hash = self._event_hash(row)
            if event_hash != expected_hash:
                return _AuditIntegrityReport(
                    False,
                    "AuditIntegrityHashMismatch",
                    latest_hash,
                    retained_event_count=len(rows),
                    quarantined_event_count=len(rows) - index,
                    quarantine_reason="tampered_or_corrupt_rows",
                )
            previous_event_hash = row.get("previous_event_hash")
            if index > 0 and previous_event_hash != previous_hash:
                return _AuditIntegrityReport(
                    False,
                    "AuditIntegrityChainMismatch",
                    latest_hash,
                    retained_event_count=len(rows),
                    quarantined_event_count=len(rows) - index,
                    quarantine_reason="chain_break_rows",
                )
            previous_hash = event_hash
            latest_hash = event_hash
        if quarantined_event_count:
            return _AuditIntegrityReport(
                False,
                "AuditIntegrityMissingHash",
                latest_hash,
                retained_event_count=len(rows),
                quarantined_event_count=quarantined_event_count,
                quarantine_reason=quarantine_reason,
            )
        return _AuditIntegrityReport(
            True,
            None,
            latest_hash,
            retained_event_count=len(rows),
        )

    def _quarantine_export_rows(
        self,
        rows: list[dict[str, str | None]],
    ) -> list[dict[str, str | None]]:
        """Select legacy, corrupt, or chain-broken rows for quarantine."""

        quarantined_rows: list[dict[str, str | None]] = []
        previous_hash: str | None = None
        for index, row in enumerate(rows):
            event_hash = row.get("event_hash")
            if event_hash is None:
                quarantined_rows.append(
                    {
                        **row,
                        "quarantine_reason": "legacy_or_unverifiable_rows",
                    }
                )
                previous_hash = None
                continue
            row_quarantine_reason = self._row_quarantine_reason(
                row,
                event_hash=event_hash,
                previous_hash=previous_hash,
                requires_previous_hash=index > 0,
            )
            if row_quarantine_reason is not None:
                quarantined_rows.extend(
                    self._quarantine_tail(
                        rows[index:],
                        quarantine_reason=row_quarantine_reason,
                    )
                )
                break
            previous_hash = event_hash
        return quarantined_rows

    def _row_quarantine_reason(
        self,
        row: dict[str, str | None],
        *,
        event_hash: str,
        previous_hash: str | None,
        requires_previous_hash: bool,
    ) -> str | None:
        """Return the stable quarantine reason for one hashed row."""

        if event_hash != self._event_hash(row):
            return "tampered_or_corrupt_rows"
        if requires_previous_hash and row.get("previous_event_hash") != previous_hash:
            return "chain_break_rows"
        return None

    def _quarantine_tail(
        self,
        rows: list[dict[str, str | None]],
        *,
        quarantine_reason: str,
    ) -> list[dict[str, str | None]]:
        """Mark a broken row and every dependent successor for quarantine."""

        return [
            {
                **row,
                "quarantine_reason": quarantine_reason,
            }
            for row in rows
        ]

    @staticmethod
    def _quarantine_summary(rows: list[dict[str, str | None]]) -> str | None:
        """Summarize the distinct quarantine reasons in an export."""

        reasons = {
            row["quarantine_reason"] for row in rows if row.get("quarantine_reason") is not None
        }
        if not reasons:
            return None
        if len(reasons) == 1:
            return next(iter(reasons))
        return "multiple_quarantine_reasons"

    def _previous_event_hash(self) -> str | None:
        """Return the newest active-log hash, ignoring blank trailing rows."""

        if not self._path.exists():
            return None
        for line in reversed(self._path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            previous_row = json.loads(line)
            previous_hash = previous_row.get("event_hash")
            return previous_hash if isinstance(previous_hash, str) else None
        return None

    @staticmethod
    def _event_hash(row: dict[str, str | None]) -> str:
        """Return the canonical SHA-256 digest for an unsigned audit row."""

        unsigned_row = dict(row)
        unsigned_row.pop("event_hash", None)
        canonical_row = json.dumps(
            unsigned_row,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical_row).hexdigest()
