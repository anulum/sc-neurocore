# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_ecology.py

from __future__ import annotations

"""Evolutionary population-ecology tests."""
import numpy as np
from sc_neurocore.evo_substrate.ecology import (
    CoevolutionArena,
    ExtinctionDetector,
    IslandModel,
    NoveltyArchive,
)
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism

__all__ = [
    "np",
    "CoevolutionArena",
    "ExtinctionDetector",
    "IslandModel",
    "NoveltyArchive",
    "Genome",
    "Organism",
]
