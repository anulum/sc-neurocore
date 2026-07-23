# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMParameters from former test_model_spike_response.py

"""Focused suite: TestSRMParameters from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403

class TestSRMParameters:
    def test_tau_eta_controls_refractory_duration(self):
        n_fast = SpikeResponseNeuron(tau_eta=5.0)
        n_slow = SpikeResponseNeuron(tau_eta=20.0)
        s_fast = len(_run(n_fast, current=10.0, steps=5000))
        s_slow = len(_run(n_slow, current=10.0, steps=5000))
        assert s_fast > s_slow

    def test_eta_reset_controls_suppression_depth(self):
        n_shallow = SpikeResponseNeuron(eta_reset=-2.0)
        n_deep = SpikeResponseNeuron(eta_reset=-10.0)
        s_shallow = len(_run(n_shallow, current=10.0, steps=5000))
        s_deep = len(_run(n_deep, current=10.0, steps=5000))
        assert s_shallow > s_deep

    def test_threshold_controls_sensitivity(self):
        n_low = SpikeResponseNeuron(v_threshold=0.5)
        n_high = SpikeResponseNeuron(v_threshold=2.0)
        s_low = len(_run(n_low, current=10.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_low > s_high

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = SpikeResponseNeuron(dt=dt)
        for _ in range(10000):
            n.step(10.0)
        assert np.isfinite(n.v)
