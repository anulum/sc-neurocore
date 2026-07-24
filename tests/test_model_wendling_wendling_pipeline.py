# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingPipeline from former test_model_wendling.py

"""Focused suite: TestWendlingPipeline from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403


class TestWendlingPipeline:
    def test_population_creates(self):
        assert Population(WendlingNeuron, n=5, label="wend").n == 5

    def test_returns_float_not_spike(self):
        """Wendling is a neural mass model returning EEG signal (float).

        Network.step_all expects int return for spike detection.
        This is documented: neural mass models are NOT spiking neurons.
        """
        n = WendlingNeuron()
        result = n.step(220.0)
        assert isinstance(result, (float, np.floating))
