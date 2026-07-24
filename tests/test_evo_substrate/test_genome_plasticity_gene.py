# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlasticityGene from former test_genome.py

"""Focused suite: TestPlasticityGene from former test_genome.py."""

from __future__ import annotations

from tests.test_evo_substrate.genome_support import *  # noqa: F403


class TestPlasticityGene:
    def test_from_vector_clamps(self) -> None:
        v = np.zeros(6)
        pg = PlasticityGene.from_vector(v)
        assert pg.stdp_lr > 0
        assert pg.stp_u_base >= 0.01
