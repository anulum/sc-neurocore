# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_speciation.py

from __future__ import annotations

"""Evolutionary speciation and diversity tests."""
import numpy as np
import pytest
import sc_neurocore.evo_substrate.speciation as speciation_mod
from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.speciation import (
    assign_species,
    genomic_distance,
    population_diversity,
    shared_fitness,
)

__all__ = [
    "np",
    "pytest",
    "speciation_mod",
    "FitnessResult",
    "Genome",
    "Organism",
    "assign_species",
    "genomic_distance",
    "population_diversity",
    "shared_fitness",
]
