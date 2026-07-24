# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowan from former test_biophysical_neurons.py

"""Focused suite: TestWilsonCowan from former test_biophysical_neurons.py."""

from __future__ import annotations

from tests.biophysical_neurons_support import *  # noqa: F403


class TestWilsonCowan:
    def test_oscillation(self):
        from sc_neurocore.neurons.models import WilsonCowanUnit

        n = WilsonCowanUnit()
        rates = [n.step(5.0) for _ in range(200)]
        assert max(rates) > min(rates), "should oscillate"
