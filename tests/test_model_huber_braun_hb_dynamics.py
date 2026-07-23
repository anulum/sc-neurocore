# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHBDynamics from former test_model_huber_braun.py

"""Focused suite: TestHBDynamics from former test_model_huber_braun.py."""

from __future__ import annotations

from tests.model_huber_braun_support import *  # noqa: F403

class TestHBDynamics:
    def test_fires_under_drive(self):
        n = HuberBraunNeuron()
        spikes = _run(n, current=50.0, steps=10_000)
        assert len(spikes) >= 1

    def test_rate_increases_with_current(self):
        s_low = len(_run(HuberBraunNeuron(), 20.0, 10_000))
        s_high = len(_run(HuberBraunNeuron(), 100.0, 10_000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [0.0, 20.0, 50.0, 100.0])
    def test_fi_sweep(self, current: float):
        n = HuberBraunNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)
