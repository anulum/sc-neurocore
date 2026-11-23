# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronGene from former test_genome.py

"""Focused suite: TestNeuronGene from former test_genome.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.evo_substrate.genome import (
    NeuronGene,
)


class TestNeuronGene:
    """The neuron gene of a genome."""

    def test_from_vector_clamps(self) -> None:
        """Decoded neuron constants are clamped to their floors."""
        v = np.zeros(8)
        ng = NeuronGene.from_vector(v)
        assert ng.tau_fast >= 0.5
        assert ng.theta >= 0.1
