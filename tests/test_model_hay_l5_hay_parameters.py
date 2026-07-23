# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayParameters from former test_model_hay_l5.py

"""Focused suite: TestHayParameters from former test_model_hay_l5.py."""

from __future__ import annotations

from tests.model_hay_l5_support import *  # noqa: F403

class TestHayParameters:
    @pytest.mark.parametrize("g_na", [150.0, 300.0, 500.0])
    def test_g_na_sweep(self, g_na: float) -> None:
        n = HayL5PyramidalNeuron(g_na=g_na)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_s)

    @pytest.mark.parametrize("g_ca_t", [0.0, 2.0, 5.0])
    def test_g_ca_trunk_sweep(self, g_ca_t: float) -> None:
        n = HayL5PyramidalNeuron(g_ca_t=g_ca_t)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_t)

    @pytest.mark.parametrize("g_st", [0.5, 1.5, 3.0])
    def test_coupling_sweep(self, g_st: float) -> None:
        n = HayL5PyramidalNeuron(g_st=g_st)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_t)
