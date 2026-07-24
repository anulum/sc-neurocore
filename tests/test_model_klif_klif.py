# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKLIF from former test_model_klif.py

"""Focused suite: TestKLIF from former test_model_klif.py."""

from __future__ import annotations

from tests.model_klif_support import *  # noqa: F403


class TestKLIF:
    def test_fires(self):
        from sc_neurocore.neurons.models.klif import KLIFNeuron

        n = KLIFNeuron()
        assert sum(n.step(0.5) for _ in range(50)) > 0
