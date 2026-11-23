# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNoFirewall from former test_hypervisor.py

"""Focused suite: TestNoFirewall from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import (
    _region,
)
from sc_neurocore.hypervisor.hypervisor import (
    Hypervisor,
    HypervisorConfig,
)


class TestNoFirewall:
    """The hypervisor with its firewall disabled."""

    def test_firewall_disabled(self) -> None:
        """Without the firewall every access is allowed."""
        hv = Hypervisor(HypervisorConfig(enable_firewall=False))
        hv.add_region(_region(0))
        assert hv.check_access("anyone", 0xDEAD_BEEF) is True
