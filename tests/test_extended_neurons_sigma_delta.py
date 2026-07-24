# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDelta from former test_extended_neurons.py

"""Focused suite: TestSigmaDelta from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403


class TestSigmaDelta:
    def test_positive_spike(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        spikes = [n.step(0.3) for _ in range(10)]
        assert 1 in spikes

    def test_negative_spike(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        spikes = [n.step(-0.3) for _ in range(10)]
        assert -1 in spikes
