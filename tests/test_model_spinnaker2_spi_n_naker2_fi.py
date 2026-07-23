# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2FI from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2FI from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403

class TestSpiNNaker2FI:
    def test_zero_silent(self):
        n = SpiNNaker2Neuron()
        assert sum(n.step(0) for _ in range(5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [100, 300, 500, 1000]:
            n = SpiNNaker2Neuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
