# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_selection.py

from __future__ import annotations

"""Evolutionary selection and survivor-regulation tests."""
import numpy as np
from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.selection import (
    AgeRegulator,
    BloatPenalizer,
    HallOfFame,
    ParetoFront,
    TournamentSelector,
    compute_bloat,
)

__all__ = ['np', 'FitnessResult', 'Genome', 'Organism', 'AgeRegulator', 'BloatPenalizer', 'HallOfFame', 'ParetoFront', 'TournamentSelector', 'compute_bloat']
