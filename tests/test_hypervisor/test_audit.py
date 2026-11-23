# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor compliance-audit contracts

"""Verify bounded structured audit behavior and historical object identity."""

from __future__ import annotations

import ast
from pathlib import Path

from sc_neurocore.hypervisor import audit
from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor.audit import (
    AuditEntry,
    AuditEventType,
    SecurityAuditLog,
)


class TestSecurityAuditLog:
    """Record, filter and fingerprint compliance-audit entries."""

    def test_log_and_count(self) -> None:
        """A logged entry raises the count to one."""
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "registered"))
        assert log.count == 1

    def test_query_by_type(self) -> None:
        """A type filter returns only entries of that event type."""
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "bad access"))
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "migrated"))
        denied = log.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(denied) == 1

    def test_query_by_tenant(self) -> None:
        """A tenant filter returns only that tenant's entries."""
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "a"))
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t1", "b"))
        assert len(log.query(tenant_id="t0")) == 1

    def test_checksum(self) -> None:
        """The log checksum is a 16-character fingerprint."""
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "x"))
        cs = log.checksum()
        assert len(cs) == 16


def test_explicit_timestamp_is_preserved() -> None:
    """An explicit timestamp is kept instead of the current clock."""
    entry = AuditEntry(
        AuditEventType.ACCESS_GRANTED,
        "tenant",
        "explicit timestamp",
        timestamp_ns=42,
    )

    assert entry.timestamp_ns == 42


def test_query_without_filters_preserves_entry_order() -> None:
    """An unfiltered query returns every entry in logging order."""
    log = SecurityAuditLog()
    first = AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "first")
    second = AuditEntry(AuditEventType.MIGRATION, "t1", "second")
    log.log(first)
    log.log(second)

    assert log.query() == [first, second]


def test_bounded_log_retains_newest_entries() -> None:
    """A full bounded log drops the oldest entry and keeps the newest."""
    log = SecurityAuditLog(max_entries=1)
    first = AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "first")
    second = AuditEntry(AuditEventType.TENANT_REMOVED, "t0", "second")
    log.log(first)
    log.log(second)

    assert log.count == 1
    assert log.query() == [second]


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    """The historical hypervisor module re-exports the audit classes themselves."""
    assert compatibility_surface.AuditEventType is audit.AuditEventType
    assert compatibility_surface.AuditEntry is audit.AuditEntry
    assert compatibility_surface.SecurityAuditLog is audit.SecurityAuditLog


def test_compliance_audit_definitions_have_one_owner() -> None:
    """The audit module defines exactly its classes, and the historical module none."""
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}

    owned_names = {"AuditEventType", "AuditEntry", "SecurityAuditLog"}
    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
