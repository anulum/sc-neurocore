# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFParameters from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFParameters from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403


class TestEPropALIFParameters:
    def test_tau_a_controls_adaptation_speed(self):
        n_fast = EPropALIFNeuron(tau_a=50.0)
        n_slow = EPropALIFNeuron(tau_a=500.0)
        s_fast = len(_run(n_fast, current=0.2, steps=5000))
        s_slow = len(_run(n_slow, current=0.2, steps=5000))
        # Faster a decay → adaptation wears off quicker → more spikes
        assert s_fast > s_slow

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = EPropALIFNeuron(dt=dt)
        for _ in range(5000):
            n.step(0.2)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = EPropALIFNeuron()
            trace = [(n.step(0.2), n.v, n.a, n.e_trace) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
