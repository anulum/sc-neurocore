# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpointStore from former test_meta_plasticity.py

"""Focused suite: TestCheckpointStore from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    CheckpointStore,
    PlasticityRuleSet,
)


class TestCheckpointStore:
    """Saving and restoring plasticity checkpoints."""

    def test_save_and_count(self) -> None:
        """A saved checkpoint is counted."""
        store = CheckpointStore()
        rs = PlasticityRuleSet()
        rs.fitness = 0.8
        store.save(rs, step=100, tag="baseline")
        assert store.count == 1

    def test_restore_best(self) -> None:
        """The best checkpoint is restored by fitness."""
        store = CheckpointStore()
        rs1 = PlasticityRuleSet()
        rs1.fitness = 0.3
        rs2 = PlasticityRuleSet()
        rs2.fitness = 0.9
        store.save(rs1, step=1)
        store.save(rs2, step=2)
        best = store.restore_best()
        assert best is not None
        assert best.fitness == 0.9

    def test_restore_by_tag(self) -> None:
        """A checkpoint is restored by its tag."""
        store = CheckpointStore()
        rs = PlasticityRuleSet()
        store.save(rs, step=1, tag="task_A")
        restored = store.restore_by_tag("task_A")
        assert restored is not None

    def test_max_checkpoints(self) -> None:
        """The store keeps at most its limit."""
        store = CheckpointStore(max_checkpoints=3)
        for i in range(5):
            store.save(PlasticityRuleSet(), step=i)
        assert store.count == 3

    def test_restore_best_empty_store(self) -> None:
        """An empty store has no best checkpoint."""
        assert CheckpointStore().restore_best() is None

    def test_restore_by_tag_missing(self) -> None:
        """An unknown tag restores nothing."""
        store = CheckpointStore()
        store.save(PlasticityRuleSet(), step=1, tag="present")
        assert store.restore_by_tag("absent") is None
