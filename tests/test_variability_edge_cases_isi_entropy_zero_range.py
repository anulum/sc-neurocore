# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsiEntropyZeroRange from former test_variability_edge_cases.py

"""Focused suite: TestIsiEntropyZeroRange from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403


class TestIsiEntropyZeroRange:
    """A perfectly regular train has a single ISI value; the zero-range check
    must short-circuit to 0.0 before np.histogram, which rejects a zero range."""

    def test_regular_train_returns_zero_without_histogram_error(self):
        train = np.zeros(100, dtype=np.int8)
        train[::10] = 1
        assert isi_entropy(train) == 0.0
