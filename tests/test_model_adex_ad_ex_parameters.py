# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExParameters from former test_model_adex.py

"""Focused suite: TestAdExParameters from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403

class TestAdExParameters:
    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = AdExNeuron(dt=dt)
        for _ in range(10000):
            n.step(500.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AdExNeuron()
            trace = [(n.step(500.0), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
