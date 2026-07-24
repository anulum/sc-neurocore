# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBackpressure from former test_twinsync.py

"""Focused suite: TestBackpressure from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestBackpressure:
    def test_accept_when_empty(self):
        bp = BackpressureController(max_queue_depth=100)
        assert bp.should_accept(0) is True

    def test_reject_when_full(self):
        bp = BackpressureController(max_queue_depth=100)
        assert bp.should_accept(100) is False
        assert bp.rejected_count == 1

    def test_rejection_rate(self):
        bp = BackpressureController(max_queue_depth=1)
        bp.should_accept(0)  # accept
        bp.should_accept(1)  # reject
        assert bp.rejection_rate == 0.5

    def test_rejection_rate_no_offers(self):
        bp = BackpressureController(max_queue_depth=10)
        assert bp.rejection_rate == 0.0

    def test_is_backpressured_above_threshold(self):
        bp = BackpressureController(max_queue_depth=1)
        bp.should_accept(1)  # reject -> rejection rate 1.0
        assert bp.is_backpressured is True
