# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableParameters from former test_model_rall_cable.py

"""Focused suite: TestRallCableParameters from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403


class TestRallCableParameters:
    @pytest.mark.parametrize("n_comp", [2, 3, 5, 10])
    def test_n_comp_variations(self, n_comp: int) -> None:
        n = RallCableNeuron(n_comp=n_comp)
        assert n.v.shape == (n_comp,)
        for _ in range(1000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        n = RallCableNeuron(dt=dt, n_comp=3, g_ratio=1.0)
        for _ in range(10000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = RallCableNeuron(n_comp=3)
            trace = [(n.step(100.0), n.v[0]) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
