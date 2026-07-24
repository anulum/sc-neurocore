# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLTCDynamics from former test_model_ltc.py

"""Focused suite: TestLTCDynamics from former test_model_ltc.py."""

from __future__ import annotations

from tests.model_ltc_support import *  # noqa: F403


class TestLTCDynamics:
    def test_fires(self):
        assert len(_run(LiquidTimeConstantNeuron(), 5.0, 5000)) >= 10

    def test_subthreshold(self):
        assert len(_run(LiquidTimeConstantNeuron(), 0.01, 5000)) == 0

    def test_rate_monotonic(self):
        rates = [len(_run(LiquidTimeConstantNeuron(), I, 5000)) for I in [1.0, 5.0, 10.0]]
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = LiquidTimeConstantNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.x)
