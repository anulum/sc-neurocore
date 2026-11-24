# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIF from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIF from former test_model_e_prop_alif.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.e_prop_alif import (
    EPropALIFNeuron,
)


class TestEPropALIF:
    """The e-prop adaptive LIF neuron."""

    def test_fires(self) -> None:
        """Sustained drive makes the neuron fire."""
        n = EPropALIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation(self) -> None:
        """Adaptation builds up while the neuron fires."""
        n = EPropALIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.a != 0.0, "adaptation variable must change after spikes"
