# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAlphaAnalysis from former test_model_alpha.py

"""Focused suite: TestAlphaAnalysis from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403

class TestAlphaAnalysis:
    """Analysis toolkit works on spikes from this model."""

    def _get_binary_train(self) -> npt.NDArray[np.int8]:
        n = AlphaNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(2.5)
        return train

    def test_firing_rate(self) -> None:
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_spike_count(self) -> None:
        train = self._get_binary_train()
        assert spike_count(train) > 0
