# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimescalePartitionerEdges from former test_intelligence_timescale_partitioner.py

"""Focused suite: TestTimescalePartitionerEdges from former test_intelligence_timescale_partitioner.py."""

from __future__ import annotations

from tests.intelligence_timescale_partitioner_support import *  # noqa: F403

class TestTimescalePartitionerEdges:
    """Cover the op-count timescale heuristic and the empty-model short-circuit."""

    def test_op_heavy_equation_is_partitioned(self) -> None:
        from sc_neurocore.compiler.intelligence import partition_timescales

        # With no explicit time constants, the arithmetic ops in each expression
        # drive the heuristic; both variables land in the partition.
        p = partition_timescales({"fast": "a", "slow": "a*b*c"})
        assert set(p.fast_equations) | set(p.slow_equations) == {"fast", "slow"}

    def test_empty_equations_returns_empty_partition(self) -> None:
        from sc_neurocore.compiler.intelligence import partition_timescales

        p = partition_timescales({})
        assert p.fast_equations == {}
        assert p.slow_equations == {}
        assert p.cdc_signals == []
