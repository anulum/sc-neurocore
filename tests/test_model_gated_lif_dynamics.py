# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDynamics from former test_model_gated_lif.py

"""Focused suite: TestDynamics from former test_model_gated_lif.py."""

from __future__ import annotations

from tests.model_gated_lif_support import *  # noqa: F403

class TestDynamics:
    def test_fires_at_test_current(self):
        n = GatedLIFNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 100

    def test_rate_increases_with_current(self):
        n_low = GatedLIFNeuron()
        n_high = GatedLIFNeuron()
        s_low = len(_run(n_low, current=2.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GatedLIFNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
