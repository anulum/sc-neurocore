# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegionFault from former test_hypervisor.py

"""Focused suite: TestRegionFault from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import (
    _region,
    _tenant,
)
from sc_neurocore.hypervisor.hypervisor import (
    Hypervisor,
    RegionState,
)


class TestRegionFault:
    """Marking regions faulted and evicting their tenants."""

    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=1024, base=0x5000_0000))
        return hv

    def test_no_faults_initially(self) -> None:
        """A new hypervisor has no faulted regions."""
        hv = self._setup()
        assert hv.get_faulted_regions() == []

    def test_mark_faulted(self) -> None:
        """A faulted region is listed and marked faulted."""
        hv = self._setup()
        hv.mark_region_faulted(0)
        assert 0 in hv.get_faulted_regions()
        assert hv.regions[0].state == RegionState.FAULTED

    def test_fault_evicts_tenant(self) -> None:
        """Faulting a region deactivates its tenant and clears its region."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        assert rid is not None
        hv.mark_region_faulted(rid)
        assert hv.tenants["t0"].active is False
        assert hv.tenants["t0"].region_id is None

    def test_fault_nonexistent_region(self) -> None:
        """Faulting an unknown region fails."""
        hv = self._setup()
        assert hv.mark_region_faulted(999) is False
