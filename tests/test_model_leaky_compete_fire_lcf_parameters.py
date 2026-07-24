# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLCFParameters from former test_model_leaky_compete_fire.py

"""Focused suite: TestLCFParameters from former test_model_leaky_compete_fire.py."""

from __future__ import annotations

from tests.model_leaky_compete_fire_support import *  # noqa: F403


class TestLCFParameters:
    @pytest.mark.parametrize("w_inh", [0.0, 0.5, 2.0])
    def test_w_inh_sweep(self, w_inh: float):
        n = LeakyCompeteFireNeuron(w_inh=w_inh)
        for _ in range(1000):
            n.step(5.0)
        assert all(np.isfinite(v) for v in n.v)

    @pytest.mark.parametrize("n_units", [2, 4, 8])
    def test_n_units_sweep(self, n_units: int):
        n = LeakyCompeteFireNeuron(n_units=n_units)
        result = n.step(5.0)
        assert len(result) == n_units
