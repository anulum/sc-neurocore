# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNOscillatoryBand from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNOscillatoryBand from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

from tests.model_fitzhugh_nagumo_support import *  # noqa: F403

class TestFHNOscillatoryBand:
    """Hopf bifurcation: oscillation in I ∈ [~0.3, ~1.2]."""

    def test_silent_below_band(self):
        n = FitzHughNagumoNeuron()
        assert len(_run(n, current=0.0, steps=10000)) <= 1

    def test_oscillatory_in_band(self):
        for I in [0.5, 0.8, 1.0]:
            n = FitzHughNagumoNeuron()
            spikes = _run(n, current=I, steps=10000)
            assert len(spikes) == {0.5: 26, 0.8: 28, 1.0: 28}[I]

    def test_suppressed_above_band(self):
        """High I pushes FP out of oscillatory region."""
        n = FitzHughNagumoNeuron()
        spikes = _run(n, current=2.0, steps=10000)
        assert spikes == [8]

    def test_regular_isi_in_band(self):
        n = FitzHughNagumoNeuron()
        spikes = _run(n, current=0.8, steps=10000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_voltage_bounded(self):
        """FHN V stays bounded ≈ [-2, 2] (cubic nullcline)."""
        n = FitzHughNagumoNeuron()
        vs = []
        for _ in range(10000):
            n.step(0.8)
            vs.append(n.v)
        assert min(vs) > -3 and max(vs) < 3
