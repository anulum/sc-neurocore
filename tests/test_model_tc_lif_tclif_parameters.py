# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCLIFParameters from former test_model_tc_lif.py

"""Focused suite: TestTCLIFParameters from former test_model_tc_lif.py."""

from __future__ import annotations

from tests.model_tc_lif_support import *  # noqa: F403

class TestTCLIFParameters:
    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = TwoCompartmentLIFNeuron(dt=dt)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.v_s)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TwoCompartmentLIFNeuron()
            trace = [(n.step(2.0, 1.0), n.v_s, n.v_d) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
