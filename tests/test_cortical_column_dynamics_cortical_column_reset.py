# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumnReset from former test_cortical_column_dynamics.py

"""Focused suite: TestCorticalColumnReset from former test_cortical_column_dynamics.py."""

from __future__ import annotations

from tests.cortical_column_dynamics_support import *  # noqa: F403


class TestCorticalColumnReset:
    def test_reset_clears_state(self):
        col = CorticalColumn(scale=0.02, seed=42)
        for _ in range(50):
            col.step()
        col.reset_state()
        result = col.step()
        # After reset, first step with low bg should not have massive activity
        total = sum(arr.sum() for arr in result.values())
        max_possible = sum(arr.shape[0] for arr in result.values())
        assert total < max_possible, "unreasonable activity after reset"
