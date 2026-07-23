# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegionFault from former test_hypervisor.py

"""Focused suite: TestRegionFault from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import *  # noqa: F403

class TestRegionFault:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=1024, base=0x5000_0000))
        return hv

    def test_no_faults_initially(self):
        hv = self._setup()
        assert hv.get_faulted_regions() == []

    def test_mark_faulted(self):
        hv = self._setup()
        hv.mark_region_faulted(0)
        assert 0 in hv.get_faulted_regions()
        assert hv.regions[0].state == RegionState.FAULTED

    def test_fault_evicts_tenant(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        hv.mark_region_faulted(rid)
        assert hv.tenants["t0"].active is False
        assert hv.tenants["t0"].region_id is None

    def test_fault_nonexistent_region(self):
        hv = self._setup()
        assert hv.mark_region_faulted(999) is False
