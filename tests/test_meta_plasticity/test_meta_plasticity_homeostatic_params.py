# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHomeostaticParams from former test_meta_plasticity.py

"""Focused suite: TestHomeostaticParams from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    HomeostaticParams,
)


class TestHomeostaticParams:
    """Homeostatic gain adaptation."""

    def test_adapt_increases_gain(self) -> None:
        """A low rate raises the gain."""
        hp = HomeostaticParams(target_rate_hz=10.0, current_gain=1.0)
        hp.adapt(5.0)  # Below target
        assert hp.current_gain > 1.0

    def test_adapt_decreases_gain(self) -> None:
        """A high rate lowers the gain."""
        hp = HomeostaticParams(target_rate_hz=5.0, current_gain=1.0)
        hp.adapt(10.0)  # Above target
        assert hp.current_gain < 1.0

    def test_gain_bounded(self) -> None:
        """The gain stays at or below its maximum."""
        hp = HomeostaticParams(target_rate_hz=100.0, gain_adaptation_rate=1.0)
        for _ in range(100):
            hp.adapt(0.0)
        assert hp.current_gain <= 10.0
