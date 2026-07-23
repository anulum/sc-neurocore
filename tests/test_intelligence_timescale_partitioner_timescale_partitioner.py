# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimescalePartitioner from former test_intelligence_timescale_partitioner.py

"""Focused suite: TestTimescalePartitioner from former test_intelligence_timescale_partitioner.py."""

from __future__ import annotations

from tests.intelligence_timescale_partitioner_support import *  # noqa: F403

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
