# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransferEntropy from former test_spike_train_stats.py

"""Focused suite: TestTransferEntropy from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestTransferEntropy:
    def test_nonnegative(self):
        a = _poisson_train(100.0, 1.0, seed=1)
        b = _poisson_train(100.0, 1.0, seed=2)
        te = transfer_entropy(a, b, bin_size=20)
        assert te >= 0
