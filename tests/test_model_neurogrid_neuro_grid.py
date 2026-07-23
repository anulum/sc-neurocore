# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuroGrid from former test_model_neurogrid.py

"""Focused suite: TestNeuroGrid from former test_model_neurogrid.py."""

from __future__ import annotations

from tests.model_neurogrid_support import *  # noqa: F403

class TestNeuroGrid:
    def test_dynamics(self) -> None:
        from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron

        n = NeuroGridNeuron()
        for _ in range(200):
            n.step(10.0)
        assert n.v_s != n.v_d, "soma and dendrite should differ"

    def test_reset(self) -> None:
        from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron

        n = NeuroGridNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert abs(n.v_s - n.v_rest) < 1e-10
