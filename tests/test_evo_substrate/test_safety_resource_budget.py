# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResourceBudget from former test_safety.py

"""Focused suite: TestResourceBudget from former test_safety.py."""

from __future__ import annotations

from tests.test_evo_substrate.safety_support import *  # noqa: F403


class TestResourceBudget:
    def test_within_budget(self) -> None:
        rb = ResourceBudget(max_neurons=1024)
        g = Genome()
        ok, violations = rb.check(g)
        assert ok
        assert violations == []

    def test_exceeds_budget(self) -> None:
        rb = ResourceBudget(max_neurons=8)
        g = Genome()  # default 16 neurons
        ok, violations = rb.check(g)
        assert not ok
        assert len(violations) > 0

    def test_exceeds_area_budget(self) -> None:
        rb = ResourceBudget(max_neurons=1024, max_area_um2=100.0)
        g = Genome()
        g.topology.num_neurons = 32
        g.topology.bitstream_length = 64

        ok, violations = rb.check(g)

        assert not ok
        assert any(violation.startswith("area=") for violation in violations)
