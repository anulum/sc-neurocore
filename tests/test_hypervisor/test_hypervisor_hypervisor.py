# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHypervisor from former test_hypervisor.py

"""Focused suite: TestHypervisor from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import (
    _region,
    _tenant,
)
from sc_neurocore.hypervisor.hypervisor import (
    Hypervisor,
    HypervisorConfig,
    TenantState,
)


class TestHypervisor:
    """Tenant registration, allocation, firewalling and migration in the hypervisor."""

    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=2048, base=0x5000_0000))
        hv.add_region(_region(2, neurons=512, base=0x6000_0000))
        return hv

    def test_register_tenant(self) -> None:
        """A new tenant is registered."""
        hv = self._setup()
        t = _tenant()
        assert hv.register_tenant(t) is True
        assert "t0" in hv.tenants

    def test_register_duplicate(self) -> None:
        """A tenant id can be registered only once."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        assert hv.register_tenant(_tenant("t0")) is False

    def test_allocate(self) -> None:
        """Allocating a tenant activates it."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        rid = hv.allocate("t0")
        assert rid is not None
        assert hv.tenants["t0"].active is True

    def test_allocate_respects_capacity(self) -> None:
        """Allocation picks the region with room for the tenant."""
        hv = self._setup()
        t = _tenant("t0")
        t.qos.max_neurons = 2000
        hv.register_tenant(t)
        rid = hv.allocate("t0")
        # Only region 1 has 2048 neurons
        assert rid == 1

    def test_deallocate(self) -> None:
        """Deallocating a tenant deactivates it."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        assert hv.deallocate("t0") is True
        assert hv.tenants["t0"].active is False

    def test_remove_tenant(self) -> None:
        """Removing a tenant drops it from the registry."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        assert hv.remove_tenant("t0") is True
        assert "t0" not in hv.tenants

    def test_schedule(self) -> None:
        """Scheduling active tenants gives time slots."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        hv.allocate("t0")
        hv.allocate("t1")
        slots = hv.schedule(10000)
        assert len(slots) > 0

    def test_firewall_access(self) -> None:
        """A tenant reaches its own region and nothing outside it."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        assert rid is not None
        region = hv.regions[rid]
        assert hv.check_access("t0", region.axi_base_addr) is True
        assert hv.check_access("t0", 0xDEAD_BEEF) is False

    def test_firewall_cross_tenant(self) -> None:
        """A tenant cannot reach another tenant's region."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        hv.allocate("t0")
        hv.allocate("t1")
        rid0 = hv.tenants["t0"].region_id
        assert rid0 is not None
        r0 = hv.regions[rid0]
        assert hv.check_access("t1", r0.axi_base_addr) is False

    def test_migrate(self) -> None:
        """Migration moves a tenant to the free region."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        old_rid = hv.tenants["t0"].region_id
        free_rid = [r for r in hv.regions if hv.regions[r].is_free][0]
        hv.tenants["t0"].state = TenantState(lfsr_state=42)
        result = hv.migrate("t0", free_rid)
        assert result.success is True
        assert hv.tenants["t0"].region_id == free_rid

    def test_status(self) -> None:
        """The status counts regions, active tenants and free regions."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        st = hv.status()
        assert st["total_regions"] == 3
        assert st["active_tenants"] == 1
        assert st["free_regions"] == 2

    def test_tenant_report(self) -> None:
        """A tenant report names the tenant and its state."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0", name="BCI"))
        hv.allocate("t0")
        rpt = hv.tenant_report("t0")
        assert rpt is not None
        assert rpt["name"] == "BCI"
        assert rpt["active"] is True

    def test_max_tenants(self) -> None:
        """Registration stops at the tenant limit."""
        hv = Hypervisor(HypervisorConfig(max_tenants=2))
        hv.add_region(_region(0))
        hv.add_region(_region(1))
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        assert hv.register_tenant(_tenant("t2")) is False

    def test_allocate_unknown_tenant(self) -> None:
        """Allocating an unknown tenant gives no region."""
        hv = self._setup()
        assert hv.allocate("ghost") is None

    def test_allocate_no_region_fits(self) -> None:
        """Allocating a tenant no region fits gives no region."""
        hv = self._setup()
        t = _tenant("big")
        t.qos.max_neurons = 100_000  # larger than every region
        hv.register_tenant(t)
        assert hv.allocate("big") is None

    def test_migrate_unknown_tenant_not_found(self) -> None:
        """Migrating an unknown tenant fails as not found."""
        hv = self._setup()
        result = hv.migrate("ghost", 1)
        assert result.success is False
        assert result.reason == "not_found"

    def test_migrate_invalid_target_region(self) -> None:
        """Migrating to an unknown region fails as invalid."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        result = hv.migrate("t0", 999)  # no such region
        assert result.success is False
        assert result.reason == "invalid_region"

    def test_tenant_report_unknown_returns_none(self) -> None:
        """A report for an unknown tenant is None."""
        hv = self._setup()
        assert hv.tenant_report("ghost") is None
