# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayCompartments from former test_model_hay_l5.py

"""Focused suite: TestHayCompartments from former test_model_hay_l5.py."""

from __future__ import annotations

from tests.model_hay_l5_support import *  # noqa: F403


class TestHayCompartments:
    def test_somatic_input_drives_spiking(self) -> None:
        """Somatic drive produces spikes (soma may hyperpolarise post-spike)."""
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(10.0) for _ in range(2000))
        assert spikes >= 1

    def test_tuft_input_depolarises_tuft(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(500):
            n.step(0.0, current_tuft=10.0)
        assert n.v_a > -75.0

    def test_coupling_propagates_soma_to_trunk(self) -> None:
        """Somatic drive → trunk depolarises via coupling."""
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        assert n.v_t > -75.0

    def test_all_compartments_evolve(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        assert n.v_s != -75.0 and n.v_t != -75.0 and n.v_a != -75.0
