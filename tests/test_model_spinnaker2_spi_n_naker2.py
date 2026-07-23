# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2 from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2 from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403

class TestSpiNNaker2:
    def test_fires(self):
        from sc_neurocore.neurons.models.spinnaker2 import SpiNNaker2Neuron

        n = SpiNNaker2Neuron()
        assert sum(n.step(200) for _ in range(100)) > 0

    def test_fixed_point(self):
        from sc_neurocore.neurons.models.spinnaker2 import SpiNNaker2Neuron

        n = SpiNNaker2Neuron()
        n.step(100)
        assert isinstance(n.v, int)
