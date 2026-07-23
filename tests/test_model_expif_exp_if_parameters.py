# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFParameters from former test_model_expif.py

"""Focused suite: TestExpIFParameters from former test_model_expif.py."""

from __future__ import annotations

from tests.model_expif_support import *  # noqa: F403

class TestExpIFParameters:
    def test_tau_affects_rate(self) -> None:
        fast = len(_run(ExpIFNeuron(tau=5.0), current=20.0, steps=10_000))
        slow = len(_run(ExpIFNeuron(tau=40.0), current=20.0, steps=10_000))
        assert fast > slow

    def test_lower_soft_threshold_fires_more_readily(self) -> None:
        lower = len(_run(ExpIFNeuron(v_rh=-62.0), current=15.0, steps=10_000))
        higher = len(_run(ExpIFNeuron(v_rh=-55.0), current=15.0, steps=10_000))
        assert lower > higher

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float) -> None:
        neuron = ExpIFNeuron(dt=dt)
        for _ in range(10_000):
            neuron.step(20.0)
        assert math.isfinite(neuron.v)
        assert neuron.v < neuron.v_threshold

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            neuron = ExpIFNeuron()
            traces.append([(neuron.step(20.0), neuron.v) for _ in range(1000)])
        assert traces[0] == traces[1]
