# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio JSONL audit persistence

"""Persistent tamper-evident audit sink for Studio policy decisions."""

from __future__ import annotations

import json
from pathlib import Path

from sc_neurocore.studio.platform.policy_audit_integrity import _JsonlAuditIntegrityMixin
from sc_neurocore.studio.platform.policy_models import (
    _AuditIntegrityReport,
    AuditEvent,
    AuditExport,
    AuditQuarantineExport,
    AuditSinkError,
    AuditSinkStatus,
)


class JsonlAuditSink(_JsonlAuditIntegrityMixin):
    """Append-only JSONL audit sink for Studio policy decisions."""

    def __init__(
        self,
        path: Path,
        *,
        rotation_bytes: int | None = None,
        retained_files: int = 5,
    ) -> None:
        """Configure an append-only audit log and bounded rotation policy."""

        if rotation_bytes is not None and rotation_bytes <= 0:
            raise ValueError("Studio audit rotation byte limit must be positive.")
        if retained_files <= 0:
            raise ValueError("Studio retained audit file count must be positive.")
        self._path = path
        self._rotation_bytes = rotation_bytes
        self._retained_files = retained_files
        self._last_error: str | None = None

    @property
    def path(self) -> Path:
        """Return the configured JSONL audit log path."""

        return self._path

    def record(self, event: AuditEvent) -> None:
        """Append a Studio policy audit event as one JSON object."""

        try:
            preflight_error = self._preflight_error()
            if preflight_error is not None:
                self._last_error = preflight_error
                raise AuditSinkError("Studio audit append failed.")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            previous_event_hash = self._previous_event_hash()
            self._rotate_if_needed()
            event_row = self._build_row(event, previous_event_hash)
            row = json.dumps(
                event_row,
                separators=(",", ":"),
                sort_keys=True,
            )
            with self._path.open("a", encoding="utf-8") as audit_file:
                audit_file.write(f"{row}\n")
        except OSError as exc:
            self._last_error = type(exc).__name__
            raise AuditSinkError("Studio audit append failed.") from exc
        self._last_error = None

    def status(self) -> AuditSinkStatus:
        """Return status for the persistent JSONL audit sink."""

        preflight_error = self._preflight_error()
        integrity = (
            _AuditIntegrityReport(False, preflight_error, None)
            if preflight_error is not None
            else self._verify_integrity()
        )
        last_error = preflight_error or self._last_error or integrity.error
        return AuditSinkStatus(
            configured=True,
            healthy=last_error is None and integrity.verified,
            integrity_error=integrity.error,
            integrity_verified=integrity.verified,
            last_error=last_error,
            latest_event_hash=integrity.latest_event_hash,
            path_configured=True,
            quarantine_reason=integrity.quarantine_reason,
            quarantined_event_count=integrity.quarantined_event_count,
            retained_event_count=integrity.retained_event_count,
            sink_type="jsonl",
        )

    def export_recent(self, limit: int = 100) -> AuditExport:
        """Export the most recent persisted audit rows without exposing paths.

        Parameters
        ----------
        limit:
            Maximum number of audit events to include. Must be positive.

        Returns
        -------
        AuditExport
            Path-free export payload containing the newest retained rows across
            rotated JSONL files and the active audit log.

        Raises
        ------
        AuditSinkError
            If the sink location is malformed or a stored row is not a JSON
            object with scalar public values.
        """

        if limit < 1:
            raise ValueError("Audit export limit must be positive.")
        preflight_error = self._preflight_error()
        if preflight_error is not None:
            self._last_error = preflight_error
            raise AuditSinkError("Studio audit export failed.")
        rows = self._export_rows()
        integrity = self._verify_integrity()
        truncated = len(rows) > limit
        selected_rows = rows[-limit:]
        return AuditExport(
            configured=True,
            sink_type="jsonl",
            event_count=len(selected_rows),
            truncated=truncated,
            events=tuple(selected_rows),
            integrity_error=integrity.error,
            integrity_verified=integrity.verified,
            latest_event_hash=integrity.latest_event_hash,
            quarantine_reason=integrity.quarantine_reason,
            quarantined_event_count=integrity.quarantined_event_count,
            retained_event_count=integrity.retained_event_count,
        )

    def export_quarantine(self, limit: int = 100) -> AuditQuarantineExport:
        """Export quarantined retained audit rows without exposing local paths.

        Parameters
        ----------
        limit:
            Maximum number of quarantined rows to include. Must be positive.

        Returns
        -------
        AuditQuarantineExport
            Path-free export payload containing retained rows that require
            migration, quarantine, or incident review.

        Raises
        ------
        AuditSinkError
            If the sink location is malformed or a stored row is not a JSON
            object with scalar public values.
        """

        if limit < 1:
            raise ValueError("Audit quarantine export limit must be positive.")
        preflight_error = self._preflight_error()
        if preflight_error is not None:
            self._last_error = preflight_error
            raise AuditSinkError("Studio audit quarantine export failed.")
        rows = self._export_rows()
        quarantined_rows = self._quarantine_export_rows(rows)
        truncated = len(quarantined_rows) > limit
        selected_rows = quarantined_rows[-limit:]
        return AuditQuarantineExport(
            configured=True,
            sink_type="jsonl",
            event_count=len(selected_rows),
            truncated=truncated,
            events=tuple(selected_rows),
            retained_event_count=len(rows),
            quarantine_reason=self._quarantine_summary(quarantined_rows),
        )

    def _preflight_error(self) -> str | None:
        """Return a stable path error without exposing the configured path."""

        if self._path.exists() and self._path.is_dir():
            return "AuditPathIsDirectory"
        if self._path.parent.exists() and not self._path.parent.is_dir():
            return "AuditParentIsNotDirectory"
        return None

    def _build_row(
        self,
        event: AuditEvent,
        previous_event_hash: str | None,
    ) -> dict[str, str | None]:
        """Build one hash-chained JSON row from an audit event."""

        row = event.to_json_dict()
        row["previous_event_hash"] = previous_event_hash
        row["event_hash"] = self._event_hash(row)
        return row

    def _rotate_if_needed(self) -> None:
        """Rotate a full active log while enforcing the retention bound."""

        if self._rotation_bytes is None:
            return
        if not self._path.exists() or self._path.stat().st_size < self._rotation_bytes:
            return
        oldest_path = self._rotated_path(self._retained_files)
        oldest_path.unlink(missing_ok=True)
        for index in range(self._retained_files - 1, 0, -1):
            source = self._rotated_path(index)
            if source.exists():
                source.replace(self._rotated_path(index + 1))
        self._path.replace(self._rotated_path(1))

    def _rotated_path(self, index: int) -> Path:
        """Return the deterministic retained-log path for one generation."""

        return self._path.with_name(f"{self._path.name}.{index}")

    def _export_paths(self) -> tuple[Path, ...]:
        """Return existing retained logs followed by the active log."""

        rotated_paths = tuple(
            self._rotated_path(index)
            for index in range(self._retained_files, 0, -1)
            if self._rotated_path(index).exists()
        )
        return (*rotated_paths, self._path)

    def _export_rows(self) -> list[dict[str, str | None]]:
        """Load and validate all retained rows for public export."""

        rows: list[dict[str, str | None]] = []
        for parsed in self._raw_export_rows():
            rows.append(self._public_export_row(parsed))
        return rows
