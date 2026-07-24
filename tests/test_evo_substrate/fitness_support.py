# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fitness.py

from __future__ import annotations

"""Evolutionary software and FPGA fitness tests."""
from sc_neurocore.evo_substrate.fitness import (
    FitnessEvaluator,
    FitnessResult,
    HWFitnessCollector,
    HWFitnessReport,
)
from sc_neurocore.evo_substrate.genome import Genome

__all__ = ["FitnessEvaluator", "FitnessResult", "HWFitnessCollector", "HWFitnessReport", "Genome"]
