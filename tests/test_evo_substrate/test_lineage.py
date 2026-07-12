# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary lineage workflow tests

"""Evolutionary lineage workflow tests."""

from __future__ import annotations

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.replication import ReplicationEngine


class TestLineage:
    def test_lineage_recorded_on_seed(self) -> None:
        re = ReplicationEngine()
        re.seed(Genome())
        assert re.lineage.num_records == 1

    def test_lineage_recorded_on_replicate(self) -> None:
        re = ReplicationEngine()
        parent = re.seed(Genome())
        re.replicate(parent)
        assert re.lineage.num_records == 2

    def test_get_ancestors(self) -> None:
        re = ReplicationEngine()
        parent = re.seed(Genome())
        child = re.replicate(parent)
        assert child is not None
        chain = re.lineage.get_ancestors(child.genome.genome_id)
        assert len(chain) >= 1


# ── Elitism Tests ───────────────────────────────────────────────────
