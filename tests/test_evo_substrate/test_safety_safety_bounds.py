# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSafetyBounds from former test_safety.py

"""Focused suite: TestSafetyBounds from former test_safety.py."""

from __future__ import annotations

from tests.test_evo_substrate.safety_support import *  # noqa: F403

class TestSafetyBounds:
    def test_clamp(self) -> None:
        sb = SafetyBounds(max_neurons=64)
        g = Genome()
        g.topology.num_neurons = 999
        sb.clamp(g)
        assert g.topology.num_neurons == 64

    def test_within_bounds(self) -> None:
        sb = SafetyBounds()
        g = Genome()
        assert sb.is_within_bounds(g)

    def test_out_of_bounds(self) -> None:
        sb = SafetyBounds(max_neurons=10)
        g = Genome()
        g.topology.num_neurons = 100
        assert not sb.is_within_bounds(g)
