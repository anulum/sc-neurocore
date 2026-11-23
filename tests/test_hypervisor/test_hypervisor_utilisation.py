# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUtilisation from former test_hypervisor.py

"""Focused suite: TestUtilisation from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import (
    _region,
    _tenant,
)
from sc_neurocore.hypervisor.hypervisor import (
    Hypervisor,
    RegionState,
)


class TestUtilisation:
    """Region utilisation reported by the hypervisor."""

    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=2048, base=0x5000_0000))
        return hv

    def test_utilisation_empty(self) -> None:
        """Empty regions report zero utilisation."""
        hv = self._setup()
        util = hv.compute_utilisation()
        assert util[0] == 0.0
        assert util[1] == 0.0

    def test_utilisation_allocated(self) -> None:
        """An allocated region reports positive utilisation."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        util = hv.compute_utilisation()
        rid = hv.tenants["t0"].region_id
        assert rid is not None
        assert util[rid] > 0.0

    def test_utilisation_bounded(self) -> None:
        """Every utilisation lies between zero and one."""
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        util = hv.compute_utilisation()
        for v in util.values():
            assert 0.0 <= v <= 1.0

    def test_utilisation_unassigned_non_free_region_is_idle(self) -> None:
        """A busy region without a tenant reports zero utilisation."""
        # A region that is not free yet carries no tenant_id (e.g. faulted out of
        # service) contributes zero utilisation.
        hv = self._setup()
        hv.regions[0].state = RegionState.FAULTED
        util = hv.compute_utilisation()
        assert util[0] == 0.0

    def test_utilisation_orphaned_region_is_full(self) -> None:
        """A region whose tenant is gone reports full utilisation."""
        # A region still tagged with a tenant_id whose tenant record is gone
        # (e.g. removed without deallocation) is treated as fully utilised.
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        assert rid is not None
        del hv.tenants["t0"]  # region keeps its tenant_id, tenant lookup misses
        util = hv.compute_utilisation()
        assert util[rid] == 1.0
