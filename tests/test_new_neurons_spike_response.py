# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeResponse from former test_new_neurons.py

"""Focused suite: TestSpikeResponse from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestSpikeResponse:
    """The spike-response neuron."""

    def test_fires_with_input(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import SpikeResponseNeuron

        n = SpikeResponseNeuron(v_threshold=0.5, tau_kappa=1.0)
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_refractory_suppression(self) -> None:
        """After a forced spike, the time since the last spike is under 2."""
        from sc_neurocore.neurons.models import SpikeResponseNeuron

        n = SpikeResponseNeuron(eta_reset=-10.0, tau_eta=5.0)
        n.step(10.0)  # force spike
        # Immediately after spike, refractory should suppress
        assert n.time_since_spike < 2.0
