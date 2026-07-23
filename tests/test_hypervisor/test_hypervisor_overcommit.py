# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOvercommit from former test_hypervisor.py

"""Focused suite: TestOvercommit from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import *  # noqa: F403

class TestOvercommit:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=512))
        return hv

    def test_no_overcommit(self):
        hv = self._setup()
        t = _tenant("t0")
        t.qos.max_neurons = 256
        hv.register_tenant(t)
        hv.allocate("t0")
        assert hv.check_overcommit() is False

    def test_overcommit_detected(self):
        hv = self._setup()
        for i in range(3):
            t = _tenant(f"t{i}")
            t.qos.max_neurons = 512
            hv.register_tenant(t)
            t.active = True
        assert hv.check_overcommit() is True
