# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor compliance audit trail

"""Record, filter, retain, and checksum structured hypervisor audit events."""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional


class AuditEventType(Enum):
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    MIGRATION = "migration"
    PREEMPTION = "preemption"
    TENANT_REGISTERED = "tenant_registered"
    TENANT_REMOVED = "tenant_removed"
    SLA_VIOLATION = "sla_violation"


@dataclass
class AuditEntry:
    event_type: AuditEventType
    tenant_id: str
    details: str
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()


class SecurityAuditLog:
    """Structured, append-only audit trail for compliance."""

    def __init__(self, max_entries: int = 10000) -> None:
        self.entries: Deque[AuditEntry] = deque(maxlen=max_entries)

    def log(self, event: AuditEntry) -> None:
        self.entries.append(event)

    def query(
        self, event_type: Optional[AuditEventType] = None, tenant_id: Optional[str] = None
    ) -> List[AuditEntry]:
        results = list(self.entries)
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if tenant_id is not None:
            results = [e for e in results if e.tenant_id == tenant_id]
        return results

    @property
    def count(self) -> int:
        return len(self.entries)

    def checksum(self) -> str:
        h = hashlib.sha256()
        for entry in self.entries:
            h.update(f"{entry.event_type.value}:{entry.tenant_id}:{entry.timestamp_ns}".encode())
        return h.hexdigest()[:16]
