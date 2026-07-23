# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExFI from former test_model_adex.py

"""Focused suite: TestAdExFI from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403

class TestAdExFI:
    def test_subthreshold_silent(self):
        n = AdExNeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [200.0, 500.0, 1000.0, 2000.0]:
            n = AdExNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
