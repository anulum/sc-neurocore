# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_chiplet_gen_edge_cases.py

from __future__ import annotations

"""Edge-case tests targeting the 12 lines pytest --cov flagged
uncovered in `chiplet_gen.py` after the multi-lang KL refine work
landed. Every test has a single, well-defined target line; comments
on each test class identify the line range it pins.

The defensive guards covered here:
  - LFSR seed clamps (`seed = 1` when `(0xACE1 + die_id*7919) & 0xFFFF == 0`)
  - ClockDomainCrossing.ratio with zero divisor
  - compute_cdc_configs / _build_conductance_matrix `continue` paths
    when topology references missing dies
  - simulate_thermal `die_state[die_id]` override branch
  - find_route_with_congestion fallback to no-exclusion BFS
  - find_route_min_bandwidth visited-skip + queue-append paths
  - ChipletGenerator.compile cross-die-skip path
"""
import pytest
from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    make_torus,
    simulate_thermal,
)
from sc_neurocore.chiplet.chiplet_gen import (
    CDCConfig,
    CongestionReport,
    DieThermal,
    PartitionAssignment,
    PowerDomain,
    adaptive_route,
    bandwidth_aware_route,
    compute_cdc_configs,
)
SEED_ZERO_DIE_ID = 3793

__all__ = ['pytest', 'ChipletDie', 'ChipletTopology', 'InterposerLink', 'InterposerTech', 'make_torus', 'simulate_thermal', 'CDCConfig', 'CongestionReport', 'DieThermal', 'PartitionAssignment', 'PowerDomain', 'adaptive_route', 'bandwidth_aware_route', 'compute_cdc_configs', 'SEED_ZERO_DIE_ID']
