# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReplayVerifier from former test_twinsync.py

"""Focused suite: TestReplayVerifier from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestReplayVerifier:
    def test_deterministic(self):
        rv = ReplayVerifier()
        cp = Checkpoint(0, 100, 0, lfsr_state=42)
        cp.compute_checksum()
        rv.record_run_a(cp)
        rv.record_run_b(cp)
        assert rv.is_deterministic
        assert rv.first_divergence_index is None

    def test_non_deterministic(self):
        rv = ReplayVerifier()
        cp_a = Checkpoint(0, 100, 0, lfsr_state=42)
        cp_a.compute_checksum()
        cp_b = Checkpoint(0, 100, 0, lfsr_state=99)
        cp_b.compute_checksum()
        rv.record_run_a(cp_a)
        rv.record_run_b(cp_b)
        assert not rv.is_deterministic
        assert rv.first_divergence_index == 0

    def test_empty(self):
        rv = ReplayVerifier()
        assert not rv.is_deterministic

    def test_compared_count_is_shorter_run_length(self):
        rv = ReplayVerifier()
        cp = Checkpoint(0, 100, 0, lfsr_state=1)
        cp.compute_checksum()
        rv.record_run_a(cp)
        rv.record_run_a(cp)
        rv.record_run_b(cp)
        assert rv.compared_count == 1
