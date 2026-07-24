# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFAnalysis from former test_model_plif.py

"""Focused suite: TestPLIFAnalysis from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403


class TestPLIFAnalysis:
    def test_spike_count(self):
        n = ParametricLIFNeuron(a=1.0)
        train = np.array([float(n.step(0.5)) for _ in range(500)])
        assert spike_count(train) > 10

    def test_spike_count_consistency(self):
        n = ParametricLIFNeuron(a=1.0)
        train = np.array([float(n.step(0.5)) for _ in range(500)])
        assert spike_count(train) == int(train.sum())
