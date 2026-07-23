# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanDynamics from former test_model_mckean.py

"""Focused suite: TestMcKeanDynamics from former test_model_mckean.py."""

from __future__ import annotations

from tests.model_mckean_support import *  # noqa: F403

class TestMcKeanDynamics:
    def test_silent_at_zero_input(self):
        n = McKeanNeuron()
        assert len(_run(n, current=0.0, steps=20_000)) == 0

    def test_oscillatory_in_band(self):
        for current in [0.4, 0.5, 0.6]:
            n = McKeanNeuron()
            spikes = _run(n, current=current, steps=20_000)
            assert len(spikes) >= 3, f"I={current}: only {len(spikes)} spikes"

    def test_rate_monotonic(self):
        rates = []
        for current in [0.3, 0.5, 0.7]:
            n = McKeanNeuron()
            rates.append(len(_run(n, current=current, steps=20_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_fi_sweep(self, current: float):
        n = McKeanNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_regular_isi(self):
        n = McKeanNeuron()
        spikes = _run(n, current=0.5, steps=50_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_bounded_orbit(self):
        n = McKeanNeuron()
        vs, ws = [], []
        for _ in range(20_000):
            n.step(0.5)
            vs.append(n.v)
            ws.append(n.w)
        assert min(vs) > -2 and max(vs) < 2
        assert min(ws) > -2 and max(ws) < 2

    def test_upward_crossing_only(self):
        n = McKeanNeuron()
        prev_v = n.v
        for _ in range(20_000):
            spike = n.step(0.5)
            if spike == 1:
                assert prev_v < n.v_peak
            prev_v = n.v
