# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpoint from former test_identity_substrate.py

"""Focused suite: TestCheckpoint from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403

class TestCheckpoint:
    def test_round_trip(self, tmp_path):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (50, N_CORTICAL))
        sub.run(duration=0.05, dt=0.001, stimuli_sequence=stim)

        path = str(tmp_path / "test_checkpoint.npz")
        Checkpoint.save(sub, path)
        restored = Checkpoint.load(path)

        assert restored.n_cortical == sub.n_cortical
        assert restored.n_inhibitory == sub.n_inhibitory
        assert restored.n_memory == sub.n_memory
        assert restored._total_steps == sub._total_steps
        np.testing.assert_array_almost_equal(restored.ee_weights, sub.ee_weights)
        assert len(restored.spike_history) == len(sub.spike_history)

    def test_merge_two_checkpoints(self, tmp_path):
        sub1 = _make_substrate(seed=42)
        sub1.run(duration=0.02, dt=0.001)
        p1 = str(tmp_path / "ckpt1.npz")
        Checkpoint.save(sub1, p1)

        sub2 = _make_substrate(seed=42)
        stim = np.random.default_rng(7).uniform(5, 15, (20, N_CORTICAL))
        sub2.run(duration=0.02, dt=0.001, stimuli_sequence=stim)
        p2 = str(tmp_path / "ckpt2.npz")
        Checkpoint.save(sub2, p2)

        merged = Checkpoint.merge([p1, p2])
        assert merged.n_cortical == N_CORTICAL
        assert merged._total_steps == sub1._total_steps + sub2._total_steps
