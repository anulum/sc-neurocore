# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaAnalysis from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaAnalysis from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaAnalysis:
    def test_spike_count(self):
        n = SigmaDeltaNeuron()
        train = np.array([float(max(0, n.step(0.3))) for _ in range(10000)])
        assert spike_count(train) > 100

    def test_spike_count_consistency(self):
        n = SigmaDeltaNeuron()
        train = np.array([float(max(0, n.step(0.3))) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
