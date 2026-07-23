# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueDynamics from former test_model_lapicque.py

"""Focused suite: TestLapicqueDynamics from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403

class TestLapicqueDynamics:
    def test_fires_under_drive(self):
        n = LapicqueNeuron()
        assert len(_run(n, current=20.0, steps=5000)) >= 100

    def test_subthreshold_silent(self):
        n = LapicqueNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [10.0, 20.0, 50.0]:
            n = LapicqueNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0, 50.0])
    def test_fi_sweep(self, current: float):
        n = LapicqueNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)
