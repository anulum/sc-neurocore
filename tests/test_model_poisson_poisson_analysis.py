# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonAnalysis from former test_model_poisson.py

"""Focused suite: TestPoissonAnalysis from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403

class TestPoissonAnalysis:
    def test_spike_count(self) -> None:
        n = PoissonNeuron(rate_hz=200.0)
        train = np.array([float(n.step()) for _ in range(10000)])
        count = spike_count(train)
        assert 1000 < count < 3000  # expected ~2000

    def test_spike_count_consistency(self) -> None:
        n = PoissonNeuron(rate_hz=200.0)
        train = np.array([float(n.step()) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
