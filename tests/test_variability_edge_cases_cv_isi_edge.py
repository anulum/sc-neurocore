# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCvIsiEdge from former test_variability_edge_cases.py

"""Focused suite: TestCvIsiEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403


class TestCvIsiEdge:
    def test_empty_train(self):
        assert np.isnan(cv_isi(np.zeros(100, dtype=np.int8)))

    def test_single_spike(self):
        train = np.zeros(100, dtype=np.int8)
        train[50] = 1
        assert np.isnan(cv_isi(train))

    def test_two_spikes(self):
        train = np.zeros(100, dtype=np.int8)
        train[20] = 1
        train[60] = 1
        result = cv_isi(train)
        assert np.isfinite(result) or np.isnan(result)
