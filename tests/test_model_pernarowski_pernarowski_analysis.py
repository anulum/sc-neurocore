# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiAnalysis from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiAnalysis from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403

class TestPernarowskiAnalysis:
    def test_spike_count(self):
        n = PernarowskiNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self):
        """spike_count on train equals manual sum."""
        n = PernarowskiNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        assert spike_count(train) == int(train.sum())
