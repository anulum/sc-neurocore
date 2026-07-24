# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumnBasic from former test_cortical_column_dynamics.py

"""Focused suite: TestCorticalColumnBasic from former test_cortical_column_dynamics.py."""

from __future__ import annotations

from tests.cortical_column_dynamics_support import *  # noqa: F403


class TestCorticalColumnBasic:
    def test_creation(self):
        col = CorticalColumn(scale=0.02, seed=42)
        assert col.scale == 0.02

    def test_step_returns_dict(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        assert isinstance(result, dict)

    def test_step_has_all_populations(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        assert EXPECTED_POPULATIONS.issubset(set(result.keys()))

    def test_step_output_shapes(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        for key, arr in result.items():
            assert arr.ndim == 1, f"{key} is not 1-D"
            assert arr.shape[0] > 0, f"{key} has zero length"

    def test_step_output_boolean(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.step()
        for key, arr in result.items():
            assert arr.dtype == np.bool_, f"{key} dtype is {arr.dtype}, expected bool"
