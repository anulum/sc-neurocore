# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor AXI address isolation

"""Enforce per-tenant AXI address isolation and detect range overlap."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from sc_neurocore.hypervisor.region import HWRegion


@dataclass
class FirewallRule:
    """One address-range access rule."""

    tenant_id: str
    base_addr: int
    size: int
    read_allowed: bool = True
    write_allowed: bool = True

    @property
    def end_addr(self) -> int:
        return self.base_addr + self.size


class BitstreamFirewall:
    """AXI address-range isolation preventing cross-tenant access.

    Each tenant can only access its own region's AXI address space.
    Any cross-region access is blocked and logged as a violation.
    """

    def __init__(self) -> None:
        self.rules: List[FirewallRule] = []
        self.violations: List[Dict[str, Any]] = []
        self.max_violations: int = 1000

    def add_rule(self, rule: FirewallRule) -> None:
        self.rules.append(rule)

    def remove_tenant_rules(self, tenant_id: str) -> int:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.tenant_id != tenant_id]
        return before - len(self.rules)

    def check_access(self, tenant_id: str, addr: int, is_write: bool = False) -> bool:
        """Check if a tenant can access an address."""
        for rule in self.rules:
            if rule.tenant_id != tenant_id:
                continue
            if rule.base_addr <= addr < rule.end_addr:
                if is_write and not rule.write_allowed:
                    self._log_violation(tenant_id, addr, "write_denied")
                    return False
                if not is_write and not rule.read_allowed:
                    self._log_violation(tenant_id, addr, "read_denied")
                    return False
                return True

        self._log_violation(tenant_id, addr, "no_rule")
        return False

    def _log_violation(self, tenant_id: str, addr: int, reason: str) -> None:
        if len(self.violations) < self.max_violations:
            self.violations.append(
                {
                    "tenant_id": tenant_id,
                    "addr": hex(addr),
                    "reason": reason,
                    "timestamp_ns": time.time_ns(),
                }
            )

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def clear_violations(self) -> None:
        self.violations.clear()


def verify_isolation(firewall: BitstreamFirewall, regions: Dict[int, HWRegion]) -> List[str]:
    """Verify that no two tenants share address ranges.

    Returns list of violation descriptions (empty = sound).
    """
    violations: List[str] = []
    rules_by_tenant: Dict[str, List[FirewallRule]] = {}
    for rule in firewall.rules:
        rules_by_tenant.setdefault(rule.tenant_id, []).append(rule)

    tenant_ids = list(rules_by_tenant.keys())
    for i in range(len(tenant_ids)):
        for j in range(i + 1, len(tenant_ids)):
            t1, t2 = tenant_ids[i], tenant_ids[j]
            for r1 in rules_by_tenant[t1]:
                for r2 in rules_by_tenant[t2]:
                    if r1.base_addr < r2.end_addr and r2.base_addr < r1.end_addr:
                        violations.append(
                            f"overlap: {t1}[{hex(r1.base_addr)}:{hex(r1.end_addr)}] "
                            f"& {t2}[{hex(r2.base_addr)}:{hex(r2.end_addr)}]"
                        )
    return violations
