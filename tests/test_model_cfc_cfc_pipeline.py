# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCFCPipeline from former test_model_cfc.py

"""Focused suite: TestCFCPipeline from former test_model_cfc.py."""

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
from sc_neurocore.neurons.models.cfc import (
    ClosedFormContinuousNeuron,
)


class TestCFCPipeline:
    """The ClosedFormContinuousNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(ClosedFormContinuousNeuron, n=10, label="cfc").n == 10

    def test_network_runs(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(ClosedFormContinuousNeuron, n=5, label="cfc")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # May not spike at default threshold — just verify no crash
        assert isinstance(mon.count, int)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = ClosedFormContinuousNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
