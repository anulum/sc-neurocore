# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary FPGA tile deployment tracking

"""Track organism allocation and utilisation across FPGA tiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sc_neurocore.evo_substrate.organism import Organism


@dataclass
class TileAllocation:
    """Maps an organism to a physical FPGA tile."""

    organism_id: str
    tile_id: int
    partition_id: int = 0
    deployed: bool = False
    bitstream_hash: str = ""


class TileDeploymentTracker:
    """Tracks which organisms are deployed on which FPGA tiles."""

    def __init__(self, num_tiles: int = 8) -> None:
        self.num_tiles = num_tiles
        self.allocations: Dict[int, Optional[TileAllocation]] = {i: None for i in range(num_tiles)}

    def deploy(self, organism: Organism, tile_id: int) -> TileAllocation:
        """Assign an organism to a tile and return the recorded allocation."""
        alloc = TileAllocation(
            organism_id=organism.genome.genome_id,
            tile_id=tile_id,
            deployed=True,
            bitstream_hash=organism.genome.genome_id,
        )
        self.allocations[tile_id] = alloc
        organism.tile_id = tile_id
        return alloc

    def evict(self, tile_id: int) -> None:
        """Mark a tile as free without mutating the former organism."""
        self.allocations[tile_id] = None

    @property
    def free_tiles(self) -> List[int]:
        """Return tile identifiers with no active allocation."""
        return [tid for tid, a in self.allocations.items() if a is None]

    @property
    def utilisation(self) -> float:
        """Return the fraction of tiles carrying an allocation."""
        used = sum(1 for a in self.allocations.values() if a is not None)
        return used / self.num_tiles if self.num_tiles > 0 else 0.0


__all__ = ["TileAllocation", "TileDeploymentTracker"]
