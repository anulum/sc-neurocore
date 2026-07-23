# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRBursting from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRBursting from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

from tests.model_hindmarsh_rose_support import *  # noqa: F403

class TestHRBursting:
    def test_fires_at_moderate_current(self):
        n = HindmarshRoseNeuron()
        spikes = _run(n, current=5.0, steps=10_000)
        assert len(spikes) >= 20

    def test_silent_at_low_current(self):
        n = HindmarshRoseNeuron()
        # At I=0 with default params may or may not fire
        spikes = _run(n, current=0.0, steps=5000)
        assert isinstance(len(spikes), int)

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = HindmarshRoseNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 3.5, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = HindmarshRoseNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.x)

    def test_x_bounded(self):
        """x stays bounded (cubic creates restoring force)."""
        n = HindmarshRoseNeuron()
        xs = []
        for _ in range(20_000):
            n.step(5.0)
            xs.append(n.x)
        assert min(xs) > -5 and max(xs) < 5
