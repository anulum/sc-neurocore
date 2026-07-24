# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWangBuzsaki from former test_biophysical_neurons.py

"""Focused suite: TestWangBuzsaki from former test_biophysical_neurons.py."""

from __future__ import annotations

from tests.biophysical_neurons_support import *  # noqa: F403


class TestWangBuzsaki:
    def test_fires(self):
        from sc_neurocore.neurons.models import WangBuzsakiNeuron

        n = WangBuzsakiNeuron()
        spikes = sum(n.step(1.0) for _ in range(200))
        assert spikes > 0

    def test_fast_spiking(self):
        from sc_neurocore.neurons.models import WangBuzsakiNeuron

        n = WangBuzsakiNeuron()
        spikes = sum(n.step(2.0) for _ in range(200))
        assert spikes >= 3, "fast-spiking interneuron should fire rapidly"
