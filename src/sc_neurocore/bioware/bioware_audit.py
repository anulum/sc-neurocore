# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tamper-evident biological-session audit records

"""Tamper-evident biological-session audit records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .bioware_validation import require_nonnegative, require_nonnegative_int


@dataclass
class BioAuditEntry:
    """One audit entry for a bio-hybrid session."""

    round_number: int
    timestamp_iso: str
    num_spikes: int
    num_opto_pulses: int
    latency_us: float
    health_score: float
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate one timestamped session-audit record."""
        require_nonnegative_int(self.round_number, "round_number")
        if not self.timestamp_iso or not self.timestamp_iso.strip():
            raise ValueError("timestamp_iso must not be empty")
        try:
            datetime.fromisoformat(self.timestamp_iso)
        except ValueError as exc:
            raise ValueError("timestamp_iso must be an ISO-8601 date or datetime") from exc
        require_nonnegative_int(self.num_spikes, "num_spikes")
        require_nonnegative_int(self.num_opto_pulses, "num_opto_pulses")
        require_nonnegative(self.latency_us, "latency_us")
        require_nonnegative(self.health_score, "health_score")
        if self.health_score > 1.0:
            raise ValueError("health_score must be <= 1")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")


@dataclass
class BioAuditLog:
    """Tamper-evident in-memory audit log for bio-hybrid experiments."""

    entries: List[BioAuditEntry] = field(default_factory=list)
    experiment_id: str = ""

    def __post_init__(self) -> None:
        """Validate experiment identity and strictly ordered audit entries."""
        if not isinstance(self.experiment_id, str):
            raise TypeError("experiment_id must be a string")
        if self.experiment_id and not self.experiment_id.strip():
            raise ValueError("experiment_id must not be whitespace only")
        previous_round = -1
        for entry in self.entries:
            if not isinstance(entry, BioAuditEntry):
                raise TypeError("entries must contain BioAuditEntry instances")
            if entry.round_number <= previous_round:
                raise ValueError("audit entry round numbers must increase strictly")
            previous_round = entry.round_number

    def log(self, entry: BioAuditEntry) -> None:
        """Append one audit entry to the session log.

        Parameters
        ----------
        entry:
            Timestamped closed-loop session summary to retain in append order.
        """
        if not isinstance(entry, BioAuditEntry):
            raise TypeError("entry must be a BioAuditEntry")
        if self.entries and entry.round_number <= self.entries[-1].round_number:
            raise ValueError("audit entry round numbers must increase strictly")
        self.entries.append(entry)

    @property
    def total_rounds(self) -> int:
        """Return the number of recorded audit entries.

        Returns
        -------
        int
            Count of entries currently stored in the log.
        """
        return len(self.entries)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialise audit entries to deterministic dictionaries.

        Returns
        -------
        list[dict[str, Any]]
            JSON-compatible records used by ``checksum`` and external evidence
            sinks.
        """
        return [
            {
                "round": e.round_number,
                "timestamp": e.timestamp_iso,
                "spikes": e.num_spikes,
                "opto_pulses": e.num_opto_pulses,
                "latency_us": e.latency_us,
                "health_score": e.health_score,
                "notes": e.notes,
            }
            for e in self.entries
        ]

    def checksum(self) -> str:
        """Return a cross-environment SHA-256 over identity and log contents."""
        payload = {
            "schema": "sc-neurocore.bioware-audit.v1",
            "experiment_id": self.experiment_id,
            "entries": self.to_list(),
        }
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
