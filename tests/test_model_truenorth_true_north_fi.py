# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthFI from former test_model_truenorth.py

"""Focused suite: TestTrueNorthFI from former test_model_truenorth.py."""

from __future__ import annotations

from tests.model_truenorth_support import *  # noqa: F403

class TestTrueNorthFI:
    def test_zero_input_silent(self):
        n = TrueNorthNeuron()
        assert sum(n.step(0) for _ in range(1000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [10, 20, 50, 100]:
            n = TrueNorthNeuron()
            rates.append(sum(n.step(I) for _ in range(1000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_suprathreshold_every_step(self):
        """I ≥ threshold → spike every step."""
        n = TrueNorthNeuron()
        spikes = sum(n.step(100) for _ in range(100))
        assert spikes == 100
