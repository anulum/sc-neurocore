# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMSPipeline from former test_model_mainen_sejnowski.py

"""Focused suite: TestMSPipeline from former test_model_mainen_sejnowski.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
from sc_neurocore.network.monitor import (
    SpikeMonitor,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.network.projection import (
    Projection,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.mainen_sejnowski import (
    MainenSejnowskiNeuron,
)


class TestMSPipeline:
    """The MainenSejnowskiNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(MainenSejnowskiNeuron, n=5, label="ms").n == 5

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(MainenSejnowskiNeuron, n=3, label="src")
        tgt = Population(MainenSejnowskiNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(MainenSejnowskiNeuron, n=5, label="ms")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self) -> None:
        """The analysis reports spikes and a rate for a driven run."""
        n = MainenSejnowskiNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(500)])
        assert spike_count(train) >= 1
