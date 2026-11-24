# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeFI from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeFI from former test_model_superspike_neuron.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.superspike_neuron import (
    SuperSpikeNeuron,
)
from tests.model_superspike_neuron_support import (
    _run,
)


class TestSuperSpikeFI:
    """The current-to-rate response of the SuperSpikeNeuron."""

    def test_zero_silent(self) -> None:
        """Zero current gives no spikes."""
        n = SuperSpikeNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_monotonic_fi(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [0.1, 0.2, 0.5, 1.0]:
            n = SuperSpikeNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
