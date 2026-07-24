# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCases from former test_wilson_cowan_parity.py

"""Focused suite: TestEdgeCases from former test_wilson_cowan_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_parity_support import *  # noqa: F403


class TestEdgeCases:
    def test_zero_length_workload(self):
        out = py_wilson_cowan_simulate(
            0.3,
            0.2,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.zeros(0),
        )
        assert out["e"].shape == (0,)
        assert out["e_final"] == 0.3
        assert out["i_final"] == 0.2

    def test_single_step(self):
        out = py_wilson_cowan_simulate(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.array([1.0]),
        )
        assert out["e"].shape == (1,)
        assert out["e_final"] == out["e"][0]
