# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStimulus from former test_network_basic.py

"""Focused suite: TestStimulus from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403


class TestStimulus:
    """Timed-array, Poisson and step-current stimulus sources."""

    def test_timed_array(self) -> None:
        """A timed array returns the value at each step and clamps past its end."""
        ta = TimedArray([0.0, 1.0, 2.0, 3.0], dt=0.001)
        assert ta.get_current(0) == 0.0
        assert ta.get_current(2) == 2.0
        assert ta.get_current(100) == 3.0  # clamp

    def test_poisson_input(self) -> None:
        """A Poisson input returns one current per input channel."""
        pi = PoissonInput(n=10, rate_hz=1000.0, weight=0.5, dt=0.001, seed=0)
        c = pi.get_current(0)
        assert c.shape == (10,)

    def test_step_current(self) -> None:
        """A step current is on from onset and off before onset and at offset."""
        sc = StepCurrent(onset=10, offset=20, amplitude=5.0)
        assert sc.get_current(5) == 0.0
        assert sc.get_current(15) == 5.0
        assert sc.get_current(20) == 0.0
