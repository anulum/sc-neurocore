# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNakerLIF from former test_extended_neurons.py

"""Focused suite: TestSpiNNakerLIF from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403


class TestSpiNNakerLIF:
    def test_fires(self):
        n = SpiNNakerLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_refractory(self):
        n = SpiNNakerLIFNeuron(tau_refrac=5.0)
        # Drive hard until spike
        for _ in range(50):
            if n.step(100.0):
                break
        assert n.refrac_count > 0
