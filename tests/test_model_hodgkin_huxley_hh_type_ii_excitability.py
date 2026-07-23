# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHTypeIIExcitability from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHTypeIIExcitability from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

from tests.model_hodgkin_huxley_support import *  # noqa: F403

class TestHHTypeIIExcitability:
    """HH is the canonical Type-II excitable model."""

    def test_subthreshold_silent(self):
        n = HodgkinHuxleyNeuron()
        assert len(_run(n, current=5.0, steps=5000)) <= 2

    def test_suprathreshold_fires(self):
        n = HodgkinHuxleyNeuron()
        assert len(_run(n, current=10.0, steps=5000)) >= 100

    def test_non_monotonic_fi(self):
        """Type-II: f–I peaks then declines at high current."""
        rates = {}
        for I in [10.0, 20.0, 50.0]:
            n = HodgkinHuxleyNeuron()
            rates[I] = len(_run(n, current=I, steps=5000))
        # Rate should peak at moderate I, decline at I=50
        assert rates[20.0] > rates[50.0], (
            f"f(20)={rates[20.0]}, f(50)={rates[50.0]} — expected non-monotonic"
        )

    def test_isi_regularity(self):
        """HH ISI has moderate variability from the 4-variable interaction.

        CV at I=10 measured ~0.26 — higher than simple IF models but
        still structured (not random). The variability comes from the
        interplay between fast Na (m,h) and slow K (n) gating.
        """
        n = HodgkinHuxleyNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.5, f"CV(ISI) = {cv:.4f}"
