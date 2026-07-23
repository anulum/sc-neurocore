# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeltaCheckpoint from former test_twinsync.py

"""Focused suite: TestDeltaCheckpoint from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403

class TestDeltaCheckpoint:
    def test_compute_delta(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        new = np.array([1.0, 9.0, 3.0, 7.0])
        dc = DeltaCheckpoint.compute_delta(base, new, 0, 1, 1000, 0)
        assert dc.num_changes == 2
        assert dc.size_bytes > 0

    def test_no_changes(self):
        state = np.array([1.0, 2.0, 3.0])
        dc = DeltaCheckpoint.compute_delta(state, state.copy(), 0, 1, 0, 0)
        assert dc.num_changes == 0

    def test_compression_ratio_zero_for_empty_delta(self):
        state = np.array([1.0, 2.0, 3.0])
        dc = DeltaCheckpoint.compute_delta(state, state.copy(), 0, 1, 0, 0)
        assert dc.size_bytes == 0
        assert dc.compression_ratio == 0.0

    def test_compression_ratio_nonzero_delta(self):
        base = np.array([1.0, 2.0])
        new = np.array([1.0, 9.0])
        dc = DeltaCheckpoint.compute_delta(base, new, 0, 1, 0, 0)
        assert dc.size_bytes > 0
        assert dc.compression_ratio == 1.0
