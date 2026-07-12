# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary speciation and diversity metrics

"""Measure genomic distance, species membership, diversity, and niche fitness."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import numpy as np

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism

_ec: Optional[Any]
try:
    _ec = importlib.import_module("sc_neurocore.evo_substrate.evo_substrate_core")
    _HAS_RUST_EVO = True
except ImportError:
    _ec = None
    _HAS_RUST_EVO = False


def genomic_distance(a: Genome, b: Genome) -> float:
    """Normalised L1 distance between genome vectors.

    Dispatches to the Rust ``evo_substrate_core.py_genomic_distance`` when
    the compiled extension is importable. The NumPy fallback is kept as
    the reference implementation and produces bit-exact identical values.
    """
    va, vb = a.to_vector(), b.to_vector()
    if _HAS_RUST_EVO and _ec is not None:
        return float(
            _ec.py_genomic_distance(
                np.ascontiguousarray(va, dtype=np.float64),
                np.ascontiguousarray(vb, dtype=np.float64),
            )
        )
    diffs = va - vb
    norms = np.abs(va) + np.abs(vb) + 1e-10
    return float(np.mean(np.abs(diffs) / norms))


def assign_species(
    population: List[Organism],
    threshold: float = 0.3,
) -> Dict[int, List[Organism]]:
    """Assign organisms to species by genomic distance.

    First organism of each species is the representative.
    """
    species: Dict[int, List[Organism]] = {}
    representatives: Dict[int, Genome] = {}
    next_id = 0

    for org in population:
        placed = False
        for sid, rep in representatives.items():
            if genomic_distance(org.genome, rep) < threshold:
                species[sid].append(org)
                placed = True
                break
        if not placed:
            species[next_id] = [org]
            representatives[next_id] = org.genome
            next_id += 1

    return species


# ── Diversity Metric ─────────────────────────────────────────────────


def population_diversity(population: List[Organism]) -> float:
    """Mean pairwise genomic distance (0 = clones, 1 = max diversity)."""
    if len(population) < 2:
        return 0.0
    dists = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dists.append(genomic_distance(population[i].genome, population[j].genome))
    return float(np.mean(dists))


def shared_fitness(
    organism: Organism,
    population: List[Organism],
    sigma: float = 0.3,
) -> float:
    """Shared fitness: divide by niche count to prevent species domination."""
    if organism.fitness is None:
        return 0.0
    raw = organism.fitness.composite
    niche_count = sum(
        1.0 for other in population if genomic_distance(organism.genome, other.genome) < sigma
    )
    return raw / max(1.0, niche_count)


__all__ = ["assign_species", "genomic_distance", "population_diversity", "shared_fitness"]
