# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR current-response behaviour

"""Verify Wilson-HR firing regimes across physiologically useful drive levels."""

from __future__ import annotations

from tests.model_wilson_hr_support import *


class TestWilsonHRCurrentResponse:
    def test_low_current_regime_is_subthreshold(self) -> None:
        for current in [0.0, 0.01, 0.03]:
            n = WilsonHRNeuron()
            assert len(_run(n, current=current, steps=5_000)) == 0

    def test_moderate_current_regime_stays_finite(self) -> None:
        for current in [0.07, 0.1, 0.14]:
            n = WilsonHRNeuron()
            for _ in range(5_000):
                n.step(current)
            assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_drive_evokes_repetitive_spiking(self) -> None:
        n = WilsonHRNeuron()
        spikes = _run(n, current=0.075, steps=5_000)
        assert len(spikes) >= 40

    def test_high_drive_produces_more_spikes_than_threshold_drive(self) -> None:
        n_low = WilsonHRNeuron()
        n = WilsonHRNeuron()
        low_spikes = _run(n_low, current=0.075, steps=5_000)
        high_spikes = _run(n, current=0.14, steps=5_000)
        assert len(high_spikes) > len(low_spikes)

    def test_fi_5_point_sweep(self) -> None:
        rates = {}
        for current in [0.0, 0.03, 0.07, 0.1, 0.14]:
            n = WilsonHRNeuron()
            rates[current] = len(_run(n, current=current, steps=5_000))
        assert rates[0.0] == rates[0.03] == 0
        assert rates[0.14] > rates[0.1] > rates[0.07] > rates[0.03]
