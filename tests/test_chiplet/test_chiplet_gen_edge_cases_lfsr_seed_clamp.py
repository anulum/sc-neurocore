# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLfsrSeedClamp from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestLfsrSeedClamp from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403

class TestLfsrSeedClamp:
    """The four `if seed == 0: seed = 1` guards in mesh_2d / ring /
    star / make_torus topology factories trigger when die_id=3793
    (and 3793 mod 65536 multiples). Pin each path."""

    def test_mesh_2d_hits_seed_zero(self) -> None:
        # 60×64 mesh has die ids 0..3839, including 3793.
        topo = ChipletTopology.mesh_2d(rows=60, cols=64)
        target = next(d for d in topo.dies if d.die_id == SEED_ZERO_DIE_ID)
        assert target.lfsr_seed == 1

    def test_ring_hits_seed_zero(self) -> None:
        topo = ChipletTopology.ring(n_dies=SEED_ZERO_DIE_ID + 1)
        assert topo.dies[SEED_ZERO_DIE_ID].lfsr_seed == 1

    def test_star_hits_seed_zero(self) -> None:
        topo = ChipletTopology.star(n_dies=SEED_ZERO_DIE_ID + 1)
        assert topo.dies[SEED_ZERO_DIE_ID].lfsr_seed == 1

    def test_make_torus_hits_seed_zero(self) -> None:
        # 64x60 torus includes die_id=3793.
        topo = make_torus(rows=64, cols=60)
        target = next(d for d in topo.dies if d.die_id == SEED_ZERO_DIE_ID)
        assert target.lfsr_seed == 1
