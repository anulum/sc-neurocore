# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestButera from former test_extended_neurons.py

"""Focused suite: TestButera from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403

class TestButera:
    def test_dynamics(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(100):
            n.step(5.0)
        assert n.h_nap != 0.5, "persistent Na inactivation must evolve"
