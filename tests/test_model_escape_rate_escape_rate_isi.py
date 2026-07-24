# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateISI from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateISI from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403


class TestEscapeRateISI:
    def test_isi_variability(self):
        """Stochastic → CV(ISI) > 0."""
        n = EscapeRateNeuron()
        spikes = _run(n, current=40.0, steps=100000)
        if len(spikes) >= 50:
            isis = np.diff(spikes).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv > 0.1

    def test_higher_current_shorter_isi(self):
        n30 = EscapeRateNeuron()
        n40 = EscapeRateNeuron()
        s30 = _run(n30, current=30.0, steps=50000)
        s40 = _run(n40, current=40.0, steps=50000)
        if len(s30) > 10 and len(s40) > 10:
            assert np.mean(np.diff(s40)) < np.mean(np.diff(s30))
