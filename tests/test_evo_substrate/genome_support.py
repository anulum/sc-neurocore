# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_genome.py

from __future__ import annotations

"""Evolutionary genome contract tests."""
import numpy as np
from sc_neurocore.evo_substrate.genome import (
    Genome,
    GenomeSerializer,
    NeuronGene,
    PlasticityGene,
    TopologyGene,
)

__all__ = ['np', 'Genome', 'GenomeSerializer', 'NeuronGene', 'PlasticityGene', 'TopologyGene']
