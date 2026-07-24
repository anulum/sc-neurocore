# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPartitionAssignmentCrossDieSkip from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestPartitionAssignmentCrossDieSkip from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403


class TestPartitionAssignmentCrossDieSkip:
    def test_connectivity_with_unmapped_neuron_skipped(self) -> None:
        # Dies 0+1, neurons 0/1 → die 0; neurons 2/3 → die 1.
        pa = PartitionAssignment(die_assignments={0: [0, 1], 1: [2, 3]})
        connectivity: list[tuple[int, int, int]] = [
            (0, 2, 256),  # cross-die: die 0 → die 1 ✓
            (99, 3, 256),  # 99 unmapped → continue (line 1485)
            (0, 1, 256),  # same-die: die 0 → die 0 → continue (1487)
        ]
        tables = pa.to_routing_tables(connectivity)
        # Only the cross-die route appears.
        assert 0 in tables
        assert len(tables[0].entries) == 1
        assert tables[0].entries[0].dst_die == 1
        assert tables[0].entries[0].src_neuron == 0
        assert tables[0].entries[0].dst_neuron == 2
