# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur catalogue smoke contracts

"""Focused catalogue-level source-profile checks for MihalasNieburNeuron."""

from sc_neurocore.neurons.models import MihalasNieburNeuron


def test_source_profile_fires_under_table_1_panel_c_drive() -> None:
    neuron = MihalasNieburNeuron()
    events = sum(neuron.step(0.002) for _ in range(1000))

    assert events == 5


def test_table_1_panel_m_currents_are_exercised() -> None:
    neuron = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    events = sum(neuron.step(0.002) for _ in range(500))

    assert events == 8
    assert neuron.i1 > 0.0
    assert neuron.i2 < 0.0
