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
    def test_log_and_count(self) -> None:
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "registered"))
        assert log.count == 1

    def test_query_by_type(self) -> None:
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "bad access"))
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "migrated"))
        denied = log.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(denied) == 1

    def test_query_by_tenant(self) -> None:
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "a"))
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t1", "b"))
        assert len(log.query(tenant_id="t0")) == 1

    def test_checksum(self) -> None:
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "x"))
        cs = log.checksum()
        assert len(cs) == 16


def test_explicit_timestamp_is_preserved() -> None:
    entry = AuditEntry(
        AuditEventType.ACCESS_GRANTED,
        "tenant",
        "explicit timestamp",
        timestamp_ns=42,
    )

    assert entry.timestamp_ns == 42


def test_query_without_filters_preserves_entry_order() -> None:
    log = SecurityAuditLog()
    first = AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "first")
    second = AuditEntry(AuditEventType.MIGRATION, "t1", "second")
    log.log(first)
    log.log(second)

    assert log.query() == [first, second]


def test_bounded_log_retains_newest_entries() -> None:
    log = SecurityAuditLog(max_entries=1)
    first = AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "first")
    second = AuditEntry(AuditEventType.TENANT_REMOVED, "t0", "second")
    log.log(first)
    log.log(second)

    assert log.count == 1
    assert log.query() == [second]


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.AuditEventType is audit.AuditEventType
    assert compatibility_surface.AuditEntry is audit.AuditEntry
    assert compatibility_surface.SecurityAuditLog is audit.SecurityAuditLog


def test_compliance_audit_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}

    owned_names = {"AuditEventType", "AuditEntry", "SecurityAuditLog"}
    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
    assert len(Path(audit.__file__).read_text(encoding="utf-8").splitlines()) <= 80
    assert len(Path(compatibility_surface.__file__).read_text(encoding="utf-8").splitlines()) <= 850
