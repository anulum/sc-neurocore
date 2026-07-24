# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteAdapterPipeline from former test_model_astrocyte_adapter.py

"""Focused suite: TestAstrocyteAdapterPipeline from former test_model_astrocyte_adapter.py."""

from __future__ import annotations

from tests.model_astrocyte_adapter_support import *  # noqa: F403


class TestAstrocyteAdapterPipeline:
    """Pipeline tests for population, network, projection, and analysis wiring."""

    def test_population(self) -> None:
        """Population construction accepts AstrocyteNeuron."""
        assert Population(AstrocyteNeuron, n=10, label="astro").n == 10

    def test_network_spikes(self) -> None:
        """Network drive produces observed astrocyte release events."""
        pop = Population(AstrocyteNeuron, n=10, label="astro")
        drive = PoissonInput(n=10, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """Projection wiring propagates activity between astrocyte populations."""
        src = Population(AstrocyteNeuron, n=5, label="src")
        tgt = Population(AstrocyteNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_tgt.count > 0

    def test_analysis(self) -> None:
        """Spike statistics consume the adapter's binary release trace."""
        n = AstrocyteNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.01)  # dt=0.01s per step
        assert rate > 0

    def test_deterministic(self) -> None:
        """Repeated runs with the same drive produce identical traces."""
        traces: list[list[tuple[int, float]]] = []
        for _ in range(2):
            n = AstrocyteNeuron()
            trace = [(n.step(0.5), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
