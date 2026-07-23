# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHTIntrinsic from former test_model_hill_tononi.py

"""Focused suite: TestHTIntrinsic from former test_model_hill_tononi.py."""

from __future__ import annotations

from tests.model_hill_tononi_support import *  # noqa: F403

class TestHTIntrinsic:
    def test_fires_at_zero_current(self):
        """Intrinsic oscillator — fires without external input."""
        n = HillTononiNeuron()
        spikes = _run(n, current=0.0, steps=10_000)
        assert len(spikes) >= 5

    def test_rate_monotonic(self):
        rates = []
        for I in [0.0, 2.0, 5.0]:
            n = HillTononiNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 3.0, 5.0])
    def test_fi_sweep(self, current: float):
        n = HillTononiNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v)
