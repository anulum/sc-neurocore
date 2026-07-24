# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCable from former test_model_rall_cable.py

"""Focused suite: TestRallCable from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403


class TestRallCable:
    def test_propagation(self) -> None:
        from sc_neurocore.neurons.models.rall_cable import RallCableNeuron

        n = RallCableNeuron()
        for _ in range(100):
            n.step(5.0)
        assert n.v[0] != n.v[-1], "voltage should differ across compartments"

    def test_reset(self) -> None:
        from sc_neurocore.neurons.models.rall_cable import RallCableNeuron

        n = RallCableNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert all(abs(vi - n.v_rest) < 1e-10 for vi in n.v)
