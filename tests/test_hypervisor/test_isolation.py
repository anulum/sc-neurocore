# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor AXI address-isolation contracts

"""Verify tenant address isolation and its historical import surface."""

from __future__ import annotations

import ast
from pathlib import Path

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import isolation
from sc_neurocore.hypervisor.isolation import (
    BitstreamFirewall,
    FirewallRule,
    verify_isolation,
)


class TestBitstreamFirewall:
    def test_allow_read(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x1050) is True

    def test_deny_out_of_range(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x2000) is False

    def test_deny_cross_tenant(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t1", 0x1050) is False

    def test_deny_write(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, write_allowed=False))
        assert fw.check_access("t0", 0x1050, is_write=True) is False

    def test_deny_read(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, read_allowed=False))
        assert fw.check_access("t0", 0x1050) is False

    def test_violation_logged(self) -> None:
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        assert fw.violation_count == 1

    def test_remove_rules(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        removed = fw.remove_tenant_rules("t0")
        assert removed == 1
        assert fw.check_access("t0", 0x1050) is False

    def test_clear_violations(self) -> None:
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        fw.clear_violations()
        assert fw.violation_count == 0


class TestIsolationVerification:
    def test_no_overlap(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        violations = verify_isolation(fw, {})
        assert violations == []

    def test_overlap_detected(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x200))
        fw.add_rule(FirewallRule("t1", 0x1100, 0x200))
        violations = verify_isolation(fw, {})
        assert len(violations) == 1
        assert "overlap" in violations[0]


def test_violation_log_capacity_is_bounded() -> None:
    firewall = BitstreamFirewall()
    firewall.max_violations = 1

    assert firewall.check_access("tenant", 0x1000) is False
    assert firewall.check_access("tenant", 0x2000) is False
    assert firewall.violation_count == 1


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.FirewallRule is isolation.FirewallRule
    assert compatibility_surface.BitstreamFirewall is isolation.BitstreamFirewall
    assert compatibility_surface.verify_isolation is isolation.verify_isolation


def test_address_isolation_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(isolation.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}

    assert facade_classes.isdisjoint({"FirewallRule", "BitstreamFirewall"})
    assert "verify_isolation" not in facade_functions
    assert owner_classes == {"FirewallRule", "BitstreamFirewall"}
    assert owner_functions == {"verify_isolation"}
    assert len(Path(isolation.__file__).read_text(encoding="utf-8").splitlines()) <= 120
    assert len(Path(compatibility_surface.__file__).read_text(encoding="utf-8").splitlines()) <= 900
