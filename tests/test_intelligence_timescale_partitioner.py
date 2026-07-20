# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Timescale partitioner contracts

"""Contracts for compiler timescale partitioning and edge handling."""

from __future__ import annotations


class TestTimescalePartitioner:
    """Multi-timescale ODE partitioning."""

    def test_single_timescale(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales({"v": "a + b"})
        assert len(p.fast_equations) == 1
        assert len(p.slow_equations) == 0

    def test_explicit_separation(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.fast_equations
        assert "w" in p.slow_equations
        assert p.slow_clock_div >= 2

    def test_cdc_signals(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "v * c"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.cdc_signals

    def test_all_fast(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 2.0},
        )
        assert len(p.slow_equations) == 0


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
