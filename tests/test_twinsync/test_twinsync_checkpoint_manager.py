# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpointManager from former test_twinsync.py

"""Focused suite: TestCheckpointManager from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403

class TestCheckpointManager:
    def test_save(self):
        mgr = CheckpointManager()
        cp = mgr.save(0, 1000)
        assert cp.node_id == 0
        assert cp.checksum != ""

    def test_find_rollback(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        mgr.save(0, 300)
        target = mgr.find_rollback_target(0, 250)
        assert target is not None
        assert target.virtual_time_ns == 200

    def test_find_rollback_exact(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        target = mgr.find_rollback_target(0, 200)
        assert target is not None
        assert target.virtual_time_ns == 200

    def test_find_rollback_none(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        target = mgr.find_rollback_target(0, 50)
        assert target is None

    def test_discard_after(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        mgr.save(0, 300)
        removed = mgr.discard_after(0, 200)
        assert removed == 1
        assert mgr.total_checkpoints == 2

    def test_gc_max_checkpoints(self):
        mgr = CheckpointManager(max_checkpoints=5)
        for t in range(20):
            mgr.save(0, t * 100)
        assert len(mgr.checkpoints[0]) <= 5

    def test_preserves_identity(self):
        mgr = CheckpointManager()
        cp = mgr.save(0, 1000, identity_deep=0.42)
        assert cp.identity_deep == 0.42
