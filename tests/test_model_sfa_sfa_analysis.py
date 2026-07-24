# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAAnalysis from former test_model_sfa.py

"""Focused suite: TestSFAAnalysis from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403


class TestSFAAnalysis:
    def test_spike_count(self):
        n = SFANeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self):
        n = SFANeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
