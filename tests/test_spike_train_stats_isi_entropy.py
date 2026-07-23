# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestISIEntropy from former test_spike_train_stats.py

"""Focused suite: TestISIEntropy from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestISIEntropy:
    def test_regular_low_entropy(self):
        train = np.zeros(2000, dtype=np.uint8)
        train[10::20] = 1
        h = isi_entropy(train)
        assert h < 2.0

    def test_poisson_higher(self):
        h = isi_entropy(_poisson_train(50.0, 5.0))
        assert h > 0
