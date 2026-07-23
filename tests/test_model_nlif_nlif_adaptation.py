# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFAdaptation from former test_model_nlif.py

"""Focused suite: TestNLIFAdaptation from former test_model_nlif.py."""

from __future__ import annotations

from tests.model_nlif_support import *  # noqa: F403

class TestNLIFAdaptation:
    def test_w_accumulates_during_spiking(self):
        n = NonlinearLIFNeuron()
        for _ in range(5000):
            n.step(20.0)
        assert n.w != 0.0

    def test_adaptation_reduces_rate(self):
        n = NonlinearLIFNeuron()
        s1 = sum(n.step(25.0) for _ in range(2500))
        s2 = sum(n.step(25.0) for _ in range(2500))
        # Adaptation should reduce later rate
        assert s1 >= s2
