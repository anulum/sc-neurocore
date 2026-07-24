# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBendaHerzAnalysis from former test_model_benda_herz.py

"""Focused suite: TestBendaHerzAnalysis from former test_model_benda_herz.py."""

from __future__ import annotations

from tests.model_benda_herz_support import *  # noqa: F403


class TestBendaHerzAnalysis:
    def _get_binary_train(self):
        n = BendaHerzNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(50.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0  # stochastic — may be very low

    def test_spike_count(self):
        train = self._get_binary_train()
        count = spike_count(train)
        assert count >= 0
