# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor firewall rules checked against the region model

"""Check firewall rules against region ownership, and move a region's rules between tenants."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Mapping, Sequence

if TYPE_CHECKING:
    from sc_neurocore.hypervisor.isolation import BitstreamFirewall, FirewallRule
    from sc_neurocore.hypervisor.region import HWRegion


def revoke_tenant_range(
    firewall: BitstreamFirewall, tenant_id: str, base_addr: int, end_addr: int
) -> int:
    """Remove the tenant's rules that share any address with a range.

    Parameters
    ----------
    firewall : BitstreamFirewall
        Firewall whose rules are edited in place.
    tenant_id : str
        Tenant whose rules are removed.
    base_addr, end_addr : int
        The range, ``end_addr`` excluded.

    Returns
    -------
    int
        Number of rules removed; the tenant's rules elsewhere stay.
    """
    before = len(firewall.rules)
    firewall.rules = [
        rule
        for rule in firewall.rules
        if rule.tenant_id != tenant_id
        or not (rule.base_addr < end_addr and base_addr < rule.end_addr)
    ]
    return before - len(firewall.rules)


def region_violations(rules: Sequence[FirewallRule], regions: Mapping[int, HWRegion]) -> List[str]:
    """Describe what breaks isolation in the region model and in the rules measured against it.

    Parameters
    ----------
    rules : sequence of FirewallRule
        Rules to measure against the regions.
    regions : mapping of int to HWRegion
        The region model by region id. With an empty mapping there is nothing to measure,
        and no rule is reported.

    Returns
    -------
    list of str
        In this order:

        - ``region_overlap``: two regions share an address;
        - ``empty_region``: a region has no positive ``axi_size``;
        - ``rule_outside_own_region``: a rule does not lie wholly inside a region that its
          tenant holds (``region.tenant_id``), so it reaches another tenant's region or none.
    """
    violations: List[str] = []
    ordered = sorted(regions.items())
    for index, (rid_a, a) in enumerate(ordered):
        for rid_b, b in ordered[index + 1 :]:
            if a.axi_base_addr < b.axi_end_addr and b.axi_base_addr < a.axi_end_addr:
                violations.append(
                    f"region_overlap: region {rid_a}[{hex(a.axi_base_addr)}:{hex(a.axi_end_addr)}] "
                    f"& region {rid_b}[{hex(b.axi_base_addr)}:{hex(b.axi_end_addr)}]"
                )
    violations.extend(
        f"empty_region: region {rid} has axi_size {region.axi_size}"
        for rid, region in ordered
        if region.axi_size <= 0
    )
    if regions:
        violations.extend(
            f"rule_outside_own_region: {rule.tenant_id}[{hex(rule.base_addr)}:{hex(rule.end_addr)}]"
            for rule in rules
            if not any(
                region.tenant_id == rule.tenant_id
                and region.axi_base_addr <= rule.base_addr
                and rule.end_addr <= region.axi_end_addr
                for region in regions.values()
            )
        )
    return violations
