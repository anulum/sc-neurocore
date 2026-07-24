# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumnDeterminism from former test_cortical_column_dynamics.py

"""Focused suite: TestCorticalColumnDeterminism from former test_cortical_column_dynamics.py."""

from __future__ import annotations

from tests.cortical_column_dynamics_support import *  # noqa: F403


class TestCorticalColumnDeterminism:
    def test_same_seed_same_output(self):
        dt = 0.1
        dur = 5.0

        col1 = CorticalColumn(scale=0.02, seed=42)
        r1 = col1.simulate(duration_ms=dur, dt=dt)

        col2 = CorticalColumn(scale=0.02, seed=42)
        r2 = col2.simulate(duration_ms=dur, dt=dt)

        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key], err_msg=f"{key} differs between runs")
