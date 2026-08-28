# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map analysis contracts

"""Focused suite: TestRulkovAnalysis from former test_model_rulkov_map.py."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_stats.basic import spike_count
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


class TestRulkovAnalysis:
    def test_spike_count(self) -> None:
        n = RulkovMapNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(50000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self) -> None:
        n = RulkovMapNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
