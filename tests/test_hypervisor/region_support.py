# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_region.py

from __future__ import annotations

"""Verify physical region geometry, placement, health, and ownership."""
import ast
from pathlib import Path
import pytest
from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import region as region_owner
from sc_neurocore.hypervisor.region import (
    HWRegion,
    RegionHealth,
    RegionState,
    select_region_multi_die,
)


def _region(rid: int = 0, neurons: int = 1024, base: int = 0x4000_0000) -> HWRegion:
    return HWRegion(
        region_id=rid,
        num_neurons=neurons,
        num_synapses=neurons * 16,
        axi_base_addr=base,
        axi_size=0x1000,
        die_id=0,
    )


__all__ = [
    "ast",
    "Path",
    "pytest",
    "compatibility_surface",
    "region_owner",
    "HWRegion",
    "RegionHealth",
    "RegionState",
    "select_region_multi_die",
    "_region",
]
