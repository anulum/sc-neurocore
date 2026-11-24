# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDarkForestAgent from former test_zero_coverage_a.py

"""Focused suite: TestDarkForestAgent from former test_zero_coverage_a.py."""

from __future__ import annotations


class TestDarkForestAgent:
    """The dark-forest agent research module."""

    def test_decide(self) -> None:
        """A decision is a string (the check is on the type only)."""
        from meta.fermi_game import DarkForestAgent

        f = DarkForestAgent(hostility_factor=0.9)
        result = f.decide(alien_signal_strength=0.7)
        assert isinstance(result, str)
