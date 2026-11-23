# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzel from former test_biophysical_neurons.py

"""Focused suite: TestPinskyRinzel from former test_biophysical_neurons.py."""

from __future__ import annotations


class TestPinskyRinzel:
    """The two-compartment Pinsky-Rinzel neuron."""

    def test_fires(self) -> None:
        """The neuron fires under drive."""
        from sc_neurocore.neurons.models import PinskyRinzelNeuron

        n = PinskyRinzelNeuron()
        spikes = sum(n.step(10.0) for _ in range(2000))
        assert spikes > 0

    def test_two_compartments(self) -> None:
        """Soma and dendrite reach different voltages."""
        from sc_neurocore.neurons.models import PinskyRinzelNeuron

        n = PinskyRinzelNeuron()
        for _ in range(200):
            n.step(10.0)
        assert n.v_s != n.v_d, "soma and dendrite must have different voltages"
