# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCLIFFI from former test_model_tc_lif.py

"""Focused suite: TestTCLIFFI from former test_model_tc_lif.py."""

from __future__ import annotations

from tests.model_tc_lif_support import *  # noqa: F403

class TestTCLIFFI:
    def test_zero_input_silent(self):
        n = TwoCompartmentLIFNeuron()
        assert len(_run(n, i_soma=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [1.5, 2.0, 3.0, 5.0]:
            n = TwoCompartmentLIFNeuron()
            rates.append(len(_run(n, i_soma=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
