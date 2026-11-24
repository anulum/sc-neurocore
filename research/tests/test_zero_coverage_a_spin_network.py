# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpinNetwork from former test_zero_coverage_a.py

"""Focused suite: TestSpinNetwork from former test_zero_coverage_a.py."""

from __future__ import annotations


class TestSpinNetwork:
    """The spin network research module."""

    def test_pachner_and_volume(self) -> None:
        """After a Pachner move the volume is a float (the check is on the type only)."""
        from transcendent.spacetime import SpinNetwork

        s = SpinNetwork(n_nodes=4)
        s.pachner_move_1_3(0)
        vol = s.calculate_volume()
        assert isinstance(vol, float)
