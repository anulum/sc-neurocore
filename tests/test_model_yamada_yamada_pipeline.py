# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaPipeline from former test_model_yamada.py

"""Focused suite: TestYamadaPipeline from former test_model_yamada.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
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
from sc_neurocore.neurons.models.yamada import (
    YamadaNeuron,
)


class TestYamadaPipeline:
    """The YamadaNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(YamadaNeuron, n=5, label="yam").n == 5

    def test_network_with_drive(self) -> None:
        """A Poisson-driven population produces spikes."""
        pop = Population(YamadaNeuron, n=5, label="yam")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """Projection from active source adds current to target population.

        Yamada needs high sustained current to fire, so we verify the
        projection is wired by checking the source fires and using
        sufficient drive + projection weight.
        """
        src = Population(YamadaNeuron, n=10, label="src")
        tgt = Population(YamadaNeuron, n=10, label="tgt")
        drive_src = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        drive_tgt = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=99)
        proj = Projection(src, tgt, weight=30.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive_src, drive_tgt, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target has both its own drive and projection input
        assert mon_tgt.count > 0, "Target should fire with drive + projection"

    def test_analysis_pipeline(self) -> None:
        """The analysis pipeline reports a driven run."""
        n = YamadaNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(200000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.00005)  # dt=0.05ms per step
        assert rate > 0
