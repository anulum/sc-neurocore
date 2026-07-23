# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFFI from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFFI from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403

class TestEPropALIFFI:
    def test_zero_silent(self):
        n = EPropALIFNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.1, 0.2, 0.5, 1.0]:
            n = EPropALIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
