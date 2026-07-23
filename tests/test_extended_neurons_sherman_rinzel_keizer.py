# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestShermanRinzelKeizer from former test_extended_neurons.py

"""Focused suite: TestShermanRinzelKeizer from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403

class TestShermanRinzelKeizer:
    def test_fires(self):
        n = ShermanRinzelKeizerNeuron()
        assert sum(n.step(3.0) for _ in range(500)) > 0
