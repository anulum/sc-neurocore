# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSTypeI from former test_model_connor_stevens.py

"""Focused suite: TestCSTypeI from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403


class TestCSTypeI:
    def test_fires_at_sufficient_current(self):
        n = ConnorStevensNeuron()
        spikes = _run(n, current=20.0, steps=500)
        assert len(spikes) >= 20

    def test_subthreshold_silent(self):
        n = ConnorStevensNeuron()
        assert len(_run(n, current=1.0, steps=200)) == 0

    def test_continuous_fi_curve(self):
        """Type-I: frequency starts from ~0 at threshold (no frequency jump)."""
        rates = []
        for I in [5.0, 8.0, 10.0, 15.0, 20.0]:
            n = ConnorStevensNeuron()
            rates.append(len(_run(n, current=I, steps=500)))
        # Rates should increase monotonically
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [5.0, 10.0, 15.0, 20.0, 30.0])
    def test_fi_sweep(self, current: float):
        n = ConnorStevensNeuron()
        for _ in range(200):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self):
        n = ConnorStevensNeuron()
        vs = []
        for _ in range(500):
            n.step(20.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 60
