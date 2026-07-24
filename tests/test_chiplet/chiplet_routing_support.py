# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_chiplet_routing.py

from __future__ import annotations

"""Routing-table, path, timing, congestion, and energy tests."""
import pytest
from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    CongestionReport,
    InterposerLink,
    InterposerTech,
    PackageEnergyReport,
    RoutingTable,
    adaptive_route,
    bandwidth_aware_route,
    compute_decorrelation_seeds,
    estimate_congestion,
    estimate_package_energy,
    find_disjoint_paths,
    link_energy_pj,
    make_torus,
    simulate_timing,
)

__all__ = [
    "pytest",
    "ChipletDie",
    "ChipletTopology",
    "CongestionReport",
    "InterposerLink",
    "InterposerTech",
    "PackageEnergyReport",
    "RoutingTable",
    "adaptive_route",
    "bandwidth_aware_route",
    "compute_decorrelation_seeds",
    "estimate_congestion",
    "estimate_package_energy",
    "find_disjoint_paths",
    "link_energy_pj",
    "make_torus",
    "simulate_timing",
]
