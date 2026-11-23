# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVectorClock from former test_twinsync.py

"""Focused suite: TestVectorClock from former test_twinsync.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.digital_twin.twinsync import (
    VectorClock,
)


class TestVectorClock:
    """Vector clock causality tracking."""

    def test_initial(self) -> None:
        """A new vector clock is all zeros."""
        vc = VectorClock(0, 3)
        assert np.all(vc.clock == 0)

    def test_tick(self) -> None:
        """A tick advances only the clock's own entry."""
        vc = VectorClock(1, 3)
        vc.tick()
        assert vc.clock[1] == 1
        assert vc.clock[0] == 0

    def test_send(self) -> None:
        """Sending stamps the clock's own entry with 1."""
        vc = VectorClock(0, 2)
        ts = vc.send()
        assert ts[0] == 1

    def test_receive(self) -> None:
        """Receiving takes the elementwise maximum and ticks."""
        vc0 = VectorClock(0, 3)
        vc0.tick()
        vc1 = VectorClock(1, 3)
        vc1.tick()
        vc1.tick()
        vc0.receive(vc1.clock.copy())
        assert vc0.clock[0] == 2  # max(1,0)+1
        assert vc0.clock[1] == 2  # max(0,2)

    def test_happened_before(self) -> None:
        """A dominated clock happened before the other."""
        vc = VectorClock(0, 2)
        vc.tick()
        other = np.array([2, 1])
        assert vc.happened_before(other) is True

    def test_not_happened_before(self) -> None:
        """A clock ahead in one entry did not happen before the other."""
        vc = VectorClock(0, 2)
        vc.clock = np.array([3, 0])
        other = np.array([2, 1])
        assert vc.happened_before(other) is False

    def test_concurrent(self) -> None:
        """Clocks that each lead in one entry are concurrent."""
        vc = VectorClock(0, 2)
        vc.clock = np.array([2, 0])
        other = np.array([0, 2])
        assert vc.concurrent_with(other) is True
