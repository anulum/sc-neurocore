# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenomeSerializer from former test_genome.py

"""Focused suite: TestGenomeSerializer from former test_genome.py."""

from __future__ import annotations

from tests.test_evo_substrate.genome_support import *  # noqa: F403


class TestGenomeSerializer:
    def test_roundtrip(self) -> None:
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        g2 = GenomeSerializer.from_dict(d)
        assert g2.genome_id == g.genome_id
        np.testing.assert_array_almost_equal(g2.to_vector(), g.to_vector(), decimal=4)

    def test_dict_keys(self) -> None:
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        assert "vector" in d
        assert "genome_id" in d
