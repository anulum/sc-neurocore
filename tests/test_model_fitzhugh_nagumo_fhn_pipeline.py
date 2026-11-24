# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNPipeline from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNPipeline from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
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
from sc_neurocore.neurons.models.fitzhugh_nagumo import (
    FitzHughNagumoNeuron,
)


class TestFHNPipeline:
    """The FitzHughNagumoNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(FitzHughNagumoNeuron, n=10, label="fhn").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(FitzHughNagumoNeuron, n=10, label="fhn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(FitzHughNagumoNeuron, n=5, label="src")
        tgt = Population(FitzHughNagumoNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self) -> None:
        """The analysis pipeline reports a driven run."""
        n = FitzHughNagumoNeuron()
        train = np.array([float(n.step(0.8)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 5
        isis = isi(train, dt=0.0001)
        assert len(isis) >= 3
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
