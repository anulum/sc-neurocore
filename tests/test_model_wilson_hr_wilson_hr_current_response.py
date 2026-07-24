# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRCurrentResponse from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRCurrentResponse from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403


class TestWilsonHRCurrentResponse:
    def test_low_current_regime_is_subthreshold(self):
        for current in [0.0, 0.3, 1.0]:
            n = WilsonHRNeuron()
            assert len(_run(n, current=current, steps=5_000)) == 0

    def test_moderate_current_regime_stays_finite(self):
        for current in [0.6, 0.8, 1.0]:
            n = WilsonHRNeuron()
            for _ in range(5_000):
                n.step(current)
            assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_drive_evokes_transient_spiking(self):
        n = WilsonHRNeuron()
        spikes = _run(n, current=2.0, steps=5_000)
        assert len(spikes) >= 1

    def test_high_drive_produces_more_transient_spikes_than_threshold_drive(self):
        n_low = WilsonHRNeuron()
        n = WilsonHRNeuron()
        low_spikes = _run(n_low, current=2.0, steps=5_000)
        high_spikes = _run(n, current=10.0, steps=5_000)
        assert len(high_spikes) > len(low_spikes)

    def test_fi_5_point_sweep(self):
        rates = {}
        for current in [0.0, 0.3, 0.6, 2.0, 10.0]:
            n = WilsonHRNeuron()
            rates[current] = len(_run(n, current=current, steps=5_000))
        assert rates[0.0] == rates[0.3] == rates[0.6] == 0
        assert rates[10.0] > rates[2.0] > rates[0.6]
