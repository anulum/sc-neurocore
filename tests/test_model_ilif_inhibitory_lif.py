# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInhibitoryLIF from former test_model_ilif.py

"""Focused suite: TestInhibitoryLIF from former test_model_ilif.py."""

from __future__ import annotations

from tests.model_ilif_support import *  # noqa: F403


class TestInhibitoryLIF:
    def test_fires(self):
        from sc_neurocore.neurons.models.ilif import InhibitoryLIFNeuron

        n = InhibitoryLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_inhibition_trace(self):
        from sc_neurocore.neurons.models.ilif import InhibitoryLIFNeuron

        n = InhibitoryLIFNeuron()
        for _ in range(50):
            if n.step(50.0):
                break
        assert n.inh_trace > 0.0
