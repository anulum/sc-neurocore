# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor region ownership and firewall isolation

"""Hold each tenant to one region, and every firewall rule to the region its tenant holds.

A second allocation must not take another region. A preemption must move the firewall rule with the region, so the
victim can no longer reach the preemptor's addresses; a victim rule that reaches across the region's edge is removed
whole. The isolation check must read the region model: overlapping or
empty regions, and a rule over a region the tenant does not hold, are violations.
"""

from __future__ import annotations

import pytest

from sc_neurocore.hypervisor.hypervisor import Hypervisor, HypervisorConfig
from sc_neurocore.hypervisor.isolation import BitstreamFirewall, FirewallRule, verify_isolation
from sc_neurocore.hypervisor.preemption import PreemptionManager
from sc_neurocore.hypervisor.region import HWRegion, RegionState
from sc_neurocore.hypervisor.tenant import Tenant


def _region(rid: int, base: int, size: int = 0x1000) -> HWRegion:
    """Return a free region of 1024 neurons at an AXI base address."""
    return HWRegion(
        region_id=rid, num_neurons=1024, num_synapses=16384, axi_base_addr=base, axi_size=size
    )


def _hypervisor(*bases: int) -> Hypervisor:
    """Return a hypervisor with one free region per base address and the tenants a and b."""
    hypervisor = Hypervisor()
    for rid, base in enumerate(bases):
        hypervisor.add_region(_region(rid, base))
    hypervisor.register_tenant(Tenant(tenant_id="a", name="a"))
    hypervisor.register_tenant(Tenant(tenant_id="b", name="b"))
    return hypervisor


def test_a_second_allocation_returns_the_tenant_s_region_without_taking_another() -> None:
    """Allocating twice keeps one region and one firewall rule, and releasing frees every region."""
    hypervisor = _hypervisor(0x4000_0000, 0x4000_1000)
    first = hypervisor.allocate("a")
    assert first is not None
    assert hypervisor.allocate("a") == first
    allocated = [rid for rid, region in hypervisor.regions.items() if not region.is_free]
    assert allocated == [first]
    assert [rule.tenant_id for rule in hypervisor.firewall.rules] == ["a"]
    assert hypervisor.deallocate("a") is True
    assert all(region.is_free for region in hypervisor.regions.values())


def test_a_preemption_moves_the_firewall_rule_with_the_region() -> None:
    """After a preemption the preemptor reaches the region's addresses and the victim does not."""
    hypervisor = _hypervisor(0x4000_0000)
    assert hypervisor.allocate("a") == 0
    event = hypervisor.preempt("a", "b", cycle=7)
    assert event is not None
    assert (event.victim_id, event.preemptor_id, event.cycle) == ("a", "b", 7)
    assert hypervisor.regions[0].tenant_id == "b"
    assert hypervisor.tenants["a"].region_id is None
    assert hypervisor.tenants["b"].region_id == 0
    assert hypervisor.check_access("b", 0x4000_0010) is True
    assert hypervisor.check_access("a", 0x4000_0010) is False
    assert verify_isolation(hypervisor.firewall, hypervisor.regions) == []


def test_a_preemption_needs_a_victim_region_and_a_preemptor_without_one() -> None:
    """A victim without a region, a preemptor that holds one, or an unknown tenant changes nothing."""
    hypervisor = _hypervisor(0x4000_0000, 0x4000_1000)
    assert hypervisor.preempt("a", "b", cycle=0) is None
    assert hypervisor.allocate("a") == 0
    assert hypervisor.allocate("b") == 1
    assert hypervisor.preempt("a", "b", cycle=0) is None
    assert hypervisor.preempt("a", "absent", cycle=0) is None
    assert hypervisor.preempt("absent", "b", cycle=0) is None
    assert {tid: tenant.region_id for tid, tenant in hypervisor.tenants.items()} == {"a": 0, "b": 1}
    assert [(rule.tenant_id, rule.base_addr) for rule in hypervisor.firewall.rules] == [
        ("a", 0x4000_0000),
        ("b", 0x4000_1000),
    ]


def test_a_preemption_without_the_firewall_moves_the_region_and_no_rule() -> None:
    """With the firewall disabled, the region still changes hands and no rule is written."""
    hypervisor = Hypervisor(HypervisorConfig(enable_firewall=False))
    hypervisor.add_region(_region(0, 0x4000_0000))
    hypervisor.register_tenant(Tenant(tenant_id="a", name="a"))
    hypervisor.register_tenant(Tenant(tenant_id="b", name="b"))
    assert hypervisor.allocate("a") == 0
    assert hypervisor.preempt("a", "b", cycle=3) is not None
    assert hypervisor.regions[0].tenant_id == "b"
    assert hypervisor.firewall.rules == []


def test_a_preemption_of_a_region_the_hypervisor_does_not_hold_changes_nothing() -> None:
    """A victim whose region id names no registered region keeps it, and nothing is recorded."""
    hypervisor = _hypervisor(0x4000_0000)
    hypervisor.tenants["a"].region_id = 9
    assert hypervisor.preempt("a", "b", cycle=0) is None
    assert hypervisor.tenants["a"].region_id == 9
    assert hypervisor.preemption.events == []


def test_the_manager_revokes_only_the_victim_s_rules_over_the_region() -> None:
    """Given the firewall, the manager removes the victim's rules over the region and grants the preemptor."""
    manager = PreemptionManager()
    firewall = BitstreamFirewall()
    region = _region(0, 0x4000_0000)
    region.state, region.tenant_id = RegionState.ALLOCATED, "v"
    victim = Tenant(tenant_id="v", name="v", region_id=0, active=True)
    preemptor = Tenant(tenant_id="p", name="p")
    firewall.add_rule(FirewallRule("v", 0x4000_0000, 0x1000))
    firewall.add_rule(FirewallRule("v", 0x5000_0000, 0x1000))
    manager.preempt(victim, preemptor, region, cycle=1, firewall=firewall)
    assert [(rule.tenant_id, rule.base_addr, rule.size) for rule in firewall.rules] == [
        ("v", 0x5000_0000, 0x1000),
        ("p", 0x4000_0000, 0x1000),
    ]


@pytest.mark.parametrize(
    ("rule_base", "inside", "outside"),
    [(0x0800, 0x1200, 0x0900), (0x1800, 0x1900, 0x2100)],
    ids=["across_the_lower_edge", "across_the_upper_edge"],
)
def test_a_preemption_removes_a_victim_rule_across_the_region_s_edge_whole(
    rule_base: int, inside: int, outside: int
) -> None:
    """A victim rule that shares any address with the region goes whole, its part outside the region too."""
    manager = PreemptionManager()
    firewall = BitstreamFirewall()
    region = _region(0, 0x1000)
    region.state, region.tenant_id = RegionState.ALLOCATED, "v"
    victim = Tenant(tenant_id="v", name="v", region_id=0, active=True)
    preemptor = Tenant(tenant_id="p", name="p")
    firewall.add_rule(FirewallRule("v", rule_base, 0x1000))
    manager.preempt(victim, preemptor, region, cycle=1, firewall=firewall)
    assert [(rule.tenant_id, rule.base_addr, rule.size) for rule in firewall.rules] == [
        ("p", 0x1000, 0x1000)
    ]
    assert firewall.check_access("v", inside) is False
    assert firewall.check_access("v", outside) is False
    assert firewall.check_access("p", inside) is True


def test_the_isolation_check_reports_a_rule_outside_the_tenant_s_own_region() -> None:
    """A rule over a region another tenant holds, or over no region at all, is a violation."""
    firewall = BitstreamFirewall()
    regions = {0: _region(0, 0x4000_0000)}
    regions[0].state, regions[0].tenant_id = RegionState.ALLOCATED, "b"
    firewall.add_rule(FirewallRule("b", 0x4000_0000, 0x1000))
    firewall.add_rule(FirewallRule("a", 0x4000_0800, 0x100))
    firewall.add_rule(FirewallRule("c", 0x6000_0000, 0x100))
    violations = verify_isolation(firewall, regions)
    assert "rule_outside_own_region: a[0x40000800:0x40000900]" in violations
    assert "rule_outside_own_region: c[0x60000000:0x60000100]" in violations
    assert not any(violation.startswith("rule_outside_own_region: b[") for violation in violations)


def test_the_isolation_check_reports_a_rule_reaching_past_its_own_region() -> None:
    """A rule that starts in the tenant's region but ends beyond it is outside the region."""
    firewall = BitstreamFirewall()
    regions = {0: _region(0, 0x4000_0000)}
    regions[0].state, regions[0].tenant_id = RegionState.ALLOCATED, "a"
    firewall.add_rule(FirewallRule("a", 0x4000_0800, 0x1000))
    assert verify_isolation(firewall, regions) == [
        "rule_outside_own_region: a[0x40000800:0x40001800]"
    ]


def test_the_isolation_check_reports_overlapping_and_empty_regions() -> None:
    """Two regions sharing addresses, and a region of no size, are violations."""
    regions = {
        0: _region(0, 0x4000_0000),
        1: _region(1, 0x4000_0800),
        2: _region(2, 0x5000_0000, size=0),
    }
    assert verify_isolation(BitstreamFirewall(), regions) == [
        "region_overlap: region 0[0x40000000:0x40001000] & region 1[0x40000800:0x40001800]",
        "empty_region: region 2 has axi_size 0",
    ]
