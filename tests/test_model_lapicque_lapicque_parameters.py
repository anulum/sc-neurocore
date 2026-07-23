# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueParameters from former test_model_lapicque.py

"""Focused suite: TestLapicqueParameters from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403

class TestLapicqueParameters:
    @pytest.mark.parametrize("tau", [5.0, 20.0, 50.0])
    def test_tau_sweep(self, tau: float):
        n = LapicqueNeuron(tau=tau)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("resistance", [0.5, 1.0, 2.0])
    def test_resistance_sweep(self, resistance: float):
        n = LapicqueNeuron(resistance=resistance)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("dt", [0.1, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = LapicqueNeuron(dt=dt)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)
