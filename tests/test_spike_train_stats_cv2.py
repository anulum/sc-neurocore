# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCV2 from former test_spike_train_stats.py

"""Focused suite: TestCV2 from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestCV2:
    def test_regular_low(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        assert cv2(train) < 0.1

    def test_poisson(self):
        c = cv2(_poisson_train(50.0, 5.0))
        assert 0.3 < c < 1.5
