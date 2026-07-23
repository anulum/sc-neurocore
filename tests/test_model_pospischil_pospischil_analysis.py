# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilAnalysis from former test_model_pospischil.py

"""Focused suite: TestPospischilAnalysis from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilAnalysis:
    def test_spike_count(self):
        n = PospischilNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(50000)])
        assert spike_count(train) >= 100

    def test_spike_count_consistency(self):
        n = PospischilNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
