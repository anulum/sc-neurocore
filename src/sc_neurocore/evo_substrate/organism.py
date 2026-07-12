# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolving organism state

"""Define the mutable state carried by one evolving organism."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from sc_neurocore.evo_substrate.genome import Genome

if TYPE_CHECKING:
    from sc_neurocore.evo_substrate.fitness import FitnessResult
    from sc_neurocore.evo_substrate.safety import RuntimeFaultCheck


@dataclass
class Organism:
    """One evolving SC organism."""

    genome: Genome
    fitness: Optional[FitnessResult] = None
    alive: bool = True
    tile_id: Optional[int] = None
    birth_generation: int = 0
    lifespan_steps: int = 0
    runtime_fault_checks: List[RuntimeFaultCheck] = field(default_factory=list)


__all__ = ["Organism"]
