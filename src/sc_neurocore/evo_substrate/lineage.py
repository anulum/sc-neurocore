# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary lineage tracking

"""Record genome ancestry and traverse parent relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sc_neurocore.evo_substrate.organism import Organism


@dataclass
class LineageRecord:
    """One entry in the ancestry log."""

    genome_id: str
    parent_id: str
    generation: int
    mutation_type: str
    fitness: float = 0.0


class LineageTracker:
    """Tracks ancestry graph for all organisms."""

    def __init__(self) -> None:
        self.records: List[LineageRecord] = []
        self._by_id: Dict[str, LineageRecord] = {}

    def record(self, organism: Organism, mutation_type: str = "seed") -> None:
        """Append one ancestry record and index it by genome identifier."""
        fit = organism.fitness.composite if organism.fitness else 0.0
        rec = LineageRecord(
            genome_id=organism.genome.genome_id,
            parent_id=organism.genome.parent_id,
            generation=organism.genome.generation,
            mutation_type=mutation_type,
            fitness=fit,
        )
        self.records.append(rec)
        self._by_id[rec.genome_id] = rec

    def get_ancestors(self, genome_id: str) -> List[LineageRecord]:
        """Walk the ancestry chain to the root."""
        chain = []
        current = genome_id
        while current in self._by_id:
            rec = self._by_id[current]
            chain.append(rec)
            current = rec.parent_id
        return chain

    @property
    def num_records(self) -> int:
        """Return the number of recorded organisms."""
        return len(self.records)


__all__ = ["LineageRecord", "LineageTracker"]
