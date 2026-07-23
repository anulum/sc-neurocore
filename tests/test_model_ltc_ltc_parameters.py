# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLTCParameters from former test_model_ltc.py

"""Focused suite: TestLTCParameters from former test_model_ltc.py."""

from __future__ import annotations

from tests.model_ltc_support import *  # noqa: F403

class TestLTCParameters:
    @pytest.mark.parametrize("tau_base", [5.0, 10.0, 20.0])
    def test_tau_base_sweep(self, tau_base: float):
        n = LiquidTimeConstantNeuron(tau_base=tau_base)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("w_tau", [-1.0, -0.5, 0.0])
    def test_w_tau_sweep(self, w_tau: float):
        n = LiquidTimeConstantNeuron(w_tau=w_tau)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.x)
