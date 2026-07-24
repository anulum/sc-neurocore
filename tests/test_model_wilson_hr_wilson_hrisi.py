# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRISI from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRISI from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403


class TestWilsonHRISI:
    def test_isi_variability_at_peak(self):
        n = WilsonHRNeuron()
        spikes = _run(n, current=0.3, steps=50_000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv > 0, f"CV(ISI) should be > 0, got {cv:.4f}"
