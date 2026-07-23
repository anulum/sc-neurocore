# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFDynamics from former test_model_fractional_lif.py

"""Focused suite: TestFLIFDynamics from former test_model_fractional_lif.py."""

from __future__ import annotations

from tests.model_fractional_lif_support import *  # noqa: F403

class TestFLIFDynamics:
    def test_fires_at_sufficient_current(self):
        n = FractionalLIFNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 100

    def test_zero_input_silent(self):
        n = FractionalLIFNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_rate_increases_with_current(self):
        n5 = FractionalLIFNeuron()
        n10 = FractionalLIFNeuron()
        s5 = len(_run(n5, current=5.0, steps=5000))
        s10 = len(_run(n10, current=10.0, steps=5000))
        assert s10 >= s5

    def test_alpha_affects_dynamics(self):
        """Lower α → more memory → different subthreshold trajectory."""
        n_low = FractionalLIFNeuron(alpha=0.5)
        n_high = FractionalLIFNeuron(alpha=0.95)
        for _ in range(200):
            n_low.step(0.5)
            n_high.step(0.5)
        assert n_low.v != pytest.approx(n_high.v)
