# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariPipeline from former test_model_amari_field.py

"""Focused suite: TestAmariPipeline from former test_model_amari_field.py."""

from __future__ import annotations

from sc_neurocore.network.monitor import (
    SpikeMonitor,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.amari_field import (
    AmariNeuralField,
)


class TestAmariPipeline:
    """The Amari neural field in the network pipeline."""

    def test_population(self) -> None:
        """A population of three fields is built."""
        assert Population(AmariNeuralField, n=3, label="amari").n == 3

    def test_network_runs(self) -> None:
        """Network accepts AmariNeuralField.

        step() gets float from PoissonInput, but the model's step() expects array — Population wraps it.
        """
        pop = Population(AmariNeuralField, n=3, label="amari")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
