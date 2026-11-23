# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLamportClock from former test_twinsync.py

"""Focused suite: TestLamportClock from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    LamportClock,
)


class TestLamportClock:
    """Lamport logical clock ordering."""

    def test_initial(self) -> None:
        """A new Lamport clock reads 0."""
        lc = LamportClock()
        assert lc.time == 0

    def test_tick(self) -> None:
        """Each tick advances the clock by one."""
        lc = LamportClock()
        assert lc.tick() == 1
        assert lc.tick() == 2

    def test_send(self) -> None:
        """Sending from a fresh clock stamps time 1."""
        lc = LamportClock()
        ts = lc.send()
        assert ts == 1

    def test_receive(self) -> None:
        """Receiving takes the maximum of both times plus one."""
        lc = LamportClock()
        lc.tick()  # local = 1
        lc.receive(5)  # max(1,5)+1 = 6
        assert lc.time == 6

    def test_receive_behind(self) -> None:
        """Receiving an older time still advances by one."""
        lc = LamportClock()
        for _ in range(10):
            lc.tick()
        lc.receive(3)  # max(10,3)+1 = 11
        assert lc.time == 11
