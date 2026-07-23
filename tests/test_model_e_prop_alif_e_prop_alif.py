# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIF from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIF from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403

class TestEPropALIF:
    def test_fires(self):
        from sc_neurocore.neurons.models.e_prop_alif import EPropALIFNeuron

        n = EPropALIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation(self):
        from sc_neurocore.neurons.models.e_prop_alif import EPropALIFNeuron

        n = EPropALIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.a != 0.0, "adaptation variable must change after spikes"
