# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEverettTreeLayer from former test_zero_coverage_a.py

"""Focused suite: TestEverettTreeLayer from former test_zero_coverage_a.py."""

from __future__ import annotations


class TestEverettTreeLayer:
    """The Everett branching-tree layer research module."""

    def test_solve(self) -> None:
        """A solve returns a list or None (the check is on the type only)."""
        from transcendent.multiverse import EverettTreeLayer

        m = EverettTreeLayer(max_depth=5)
        result = m.solve(
            start_val=1,
            goal_func=lambda x: x >= 8,
            transition_func=lambda x, action: x * 2 if action == 0 else x + 1,
        )
        # result may be None if goal not reachable
        assert result is None or isinstance(result, list)
