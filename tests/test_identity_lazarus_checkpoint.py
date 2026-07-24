# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpoint from former test_identity_lazarus.py

"""Focused suite: TestCheckpoint from former test_identity_lazarus.py."""

from __future__ import annotations

from tests.identity_lazarus_support import *  # noqa: F403


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.02, dt=0.001)
        state_before = sub.extract_state()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            restored = Checkpoint.load(path)
            state_after = restored.extract_state()

            # Compare total_steps if available
            if "total_steps" in state_before and "total_steps" in state_after:
                assert state_before["total_steps"] == state_after["total_steps"]
        finally:
            os.remove(path)

    def test_merge_same_architecture(self):
        """Merge two checkpoints from same architecture, same seed."""
        sub1 = IdentitySubstrate(n_cortical=20, n_inhibitory=8, n_memory=4, seed=42)
        sub1.run(duration=0.01, dt=0.001)

        sub2 = IdentitySubstrate(n_cortical=20, n_inhibitory=8, n_memory=4, seed=42)
        sub2.run(duration=0.01, dt=0.001)

        tmpdir = tempfile.mkdtemp()
        p1 = os.path.join(tmpdir, "s1.npz")
        p2 = os.path.join(tmpdir, "s2.npz")

        try:
            Checkpoint.save(sub1, p1)
            Checkpoint.save(sub2, p2)

            merged = Checkpoint.merge([p1, p2])
            assert merged is not None
            merged_state = merged.extract_state()
            assert isinstance(merged_state, dict)
        finally:
            os.remove(p1)
            os.remove(p2)
            os.rmdir(tmpdir)

    def test_load_file_not_found(self):
        try:
            Checkpoint.load("/nonexistent/path.npz")
            raise AssertionError("should raise FileNotFoundError")
        except (FileNotFoundError, OSError):
            pass
