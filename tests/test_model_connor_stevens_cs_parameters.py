# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSParameters from former test_model_connor_stevens.py

"""Focused suite: TestCSParameters from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403


class TestCSParameters:
    @pytest.mark.parametrize("g_a", [0.0, 47.7, 100.0])
    def test_g_a_sweep(self, g_a: float):
        n = ConnorStevensNeuron(g_a=g_a)
        for _ in range(200):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_na", [60.0, 120.0, 200.0])
    def test_g_na_sweep(self, g_na: float):
        n = ConnorStevensNeuron(g_na=g_na)
        for _ in range(200):
            n.step(20.0)
        assert np.isfinite(n.v)
