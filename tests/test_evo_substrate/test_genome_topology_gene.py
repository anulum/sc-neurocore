# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopologyGene from former test_genome.py

"""Focused suite: TestTopologyGene from former test_genome.py."""

from __future__ import annotations

from tests.test_evo_substrate.genome_support import *  # noqa: F403

class TestTopologyGene:
    def test_from_vector_clamps(self) -> None:
        v = np.array([0.0, 0.0, -1.0, -1.0, 0.0])
        tg = TopologyGene.from_vector(v)
        assert tg.num_neurons >= 2
        assert tg.num_layers >= 1
        assert tg.connectivity >= 0.01
        assert tg.bitstream_length >= 32
