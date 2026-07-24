# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSessionSnapshot from former test_twinsync.py

"""Focused suite: TestSessionSnapshot from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestSessionSnapshot:
    def test_from_session(self):
        ts = TwinSession(2)
        ts.inject_physical_event(100, 0)
        ts.advance(1)
        snap = SessionSnapshot.from_session(ts)
        assert snap.num_nodes == 2
        assert snap.physical_events_in == 1

    def test_to_dict(self):
        ts = TwinSession(2)
        snap = SessionSnapshot.from_session(ts)
        d = snap.to_dict()
        assert "session_time_ns" in d
        assert "node_lvts" in d
