# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLamportClock from former test_twinsync.py

"""Focused suite: TestLamportClock from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestLamportClock:
    def test_initial(self):
        lc = LamportClock()
        assert lc.time == 0

    def test_tick(self):
        lc = LamportClock()
        assert lc.tick() == 1
        assert lc.tick() == 2

    def test_send(self):
        lc = LamportClock()
        ts = lc.send()
        assert ts == 1

    def test_receive(self):
        lc = LamportClock()
        lc.tick()  # local = 1
        lc.receive(5)  # max(1,5)+1 = 6
        assert lc.time == 6

    def test_receive_behind(self):
        lc = LamportClock()
        for _ in range(10):
            lc.tick()
        lc.receive(3)  # max(10,3)+1 = 11
        assert lc.time == 11
