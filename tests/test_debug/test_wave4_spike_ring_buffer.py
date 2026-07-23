# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeRingBuffer from former test_wave4.py

"""Focused suite: TestSpikeRingBuffer from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestSpikeRingBuffer:
    def test_push_and_snapshot(self):
        rb = SpikeRingBuffer(4)
        for i in range(3):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot()
        assert len(snap) == 3
        assert snap[0].sequence == 0

    def test_overwrite_on_full(self):
        rb = SpikeRingBuffer(2)
        for i in range(5):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot()
        assert len(snap) == 2
        assert snap[-1].sequence == 4

    def test_snapshot_limit(self):
        rb = SpikeRingBuffer(100)
        for i in range(50):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot(5)
        assert len(snap) == 5
