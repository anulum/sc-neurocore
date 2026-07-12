# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary FPGA tile deployment tests

"""Evolutionary FPGA tile deployment tests."""

from __future__ import annotations

from sc_neurocore.evo_substrate.deployment import TileDeploymentTracker
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism


class TestTileDeployment:
    def test_deploy(self) -> None:
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        org = Organism(genome=g)
        alloc = tracker.deploy(org, 0)
        assert alloc.deployed
        assert org.tile_id == 0

    def test_free_tiles(self) -> None:
        tracker = TileDeploymentTracker(num_tiles=4)
        assert len(tracker.free_tiles) == 4
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 1)
        assert len(tracker.free_tiles) == 3

    def test_evict(self) -> None:
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 0)
        tracker.evict(0)
        assert 0 in tracker.free_tiles

    def test_utilisation(self) -> None:
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 0)
        assert tracker.utilisation == 0.25


# ── Hall of Fame Tests (Gap 3) ────────────────────────────────────────
