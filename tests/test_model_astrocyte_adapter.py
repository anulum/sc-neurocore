# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AstrocyteNeuron (adapter)

"""Full pipeline test for AstrocyteNeuron (adapter wrapping AstrocyteModel).

Converts Ca²⁺ → spike: fires when Ca > ca_threshold. Population-compatible.
step() returns int {0,1}. Performance: ~86K steps/s.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.astrocyte_adapter import AstrocyteNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: AstrocyteNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestAstrocyteAdapterIsolation:
    """Isolation tests for adapter state and validation contracts."""

    def test_defaults(self) -> None:
        """Default parameters expose resting calcium as pseudo-voltage."""
        n = AstrocyteNeuron()
        assert n.ca_threshold == 0.3
        assert n.dt == 0.01
        assert n.v == n.ca  # v exposes Ca

    def test_step_returns_binary(self) -> None:
        """Adapter converts Ca to int {0,1}."""
        assert AstrocyteNeuron().step(0.0) in (0, 1)

    def test_v_tracks_ca(self) -> None:
        """V attribute mirrors Ca concentration."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.v == n.ca

    def test_ca_property(self) -> None:
        """Ca property delegates to the wrapped astrocyte model."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.ca > 0

    def test_ip3_property(self) -> None:
        """IP3 property delegates to the wrapped astrocyte model."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.ip3 > 0

    def test_state_finite(self) -> None:
        """Long adapter runs keep exposed state finite."""
        n = AstrocyteNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.ca) and np.isfinite(n.ip3)

    def test_reset(self) -> None:
        """Reset restores resting calcium as pseudo-voltage."""
        n = AstrocyteNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert n.v == 0.05  # ca initial

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ca_threshold": -0.01},
            {"ca_threshold": float("nan")},
            {"ca_threshold": float("inf")},
            {"dt": 0.0},
            {"dt": -0.01},
            {"dt": float("nan")},
            {"dt": float("inf")},
        ],
    )
    def test_rejects_non_physical_adapter_parameters(self, kwargs: dict[str, float]) -> None:
        """Adapter threshold and timestep must be finite physical parameters."""
        with pytest.raises(ValueError):
            AstrocyteNeuron(**kwargs)

    @pytest.mark.parametrize("current", [-0.01, float("nan"), float("inf")])
    def test_rejects_non_physical_adapter_drive(self, current: float) -> None:
        """Adapter must preserve the finite non-negative IP3 drive contract."""
        with pytest.raises(ValueError, match="current"):
            AstrocyteNeuron().step(current)


class TestAstrocyteAdapterSpikeConversion:
    """Tests for calcium-to-spike conversion semantics."""

    def test_fires_when_ca_above_threshold(self) -> None:
        """Spike = 1 when Ca > ca_threshold."""
        n = AstrocyteNeuron(ca_threshold=0.3)
        spikes_no_input = sum(n.step(0.0) for _ in range(10000))
        # Ca oscillates to 0.94 at I=0 → crosses 0.3 → spikes
        assert spikes_no_input > 0

    def test_ip3_input_drives_sustained_activity(self) -> None:
        """Sustained IP3 input keeps Ca high and fires almost every step."""
        n = AstrocyteNeuron()
        spikes = sum(n.step(0.5) for _ in range(10000))
        assert spikes > 9000, f"Only {spikes} spikes at I=0.5"

    def test_lower_threshold_more_spikes(self) -> None:
        """Lower thresholds produce more release events than high thresholds."""
        n_low = AstrocyteNeuron(ca_threshold=0.1)
        n_high = AstrocyteNeuron(ca_threshold=0.8)
        s_low = sum(n_low.step(0.0) for _ in range(10000))
        s_high = sum(n_high.step(0.0) for _ in range(10000))
        assert s_low > s_high

    def test_zero_input_oscillatory_spiking(self) -> None:
        """At I=0, Ca oscillates → intermittent spikes (not every step)."""
        n = AstrocyteNeuron(ca_threshold=0.3)
        outputs = [n.step(0.0) for _ in range(10000)]
        spikes = outputs.count(1)
        assert 100 < spikes < 9000, f"{spikes} — expected intermittent"


class TestAstrocyteAdapterPerformance:
    """Smoke tests for local adapter throughput budgets."""

    def test_isolation_throughput(self) -> None:
        """Single-adapter stepping stays above the local smoke threshold."""
        n = AstrocyteNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000

    def test_network_throughput(self) -> None:
        """Network execution stays above the local smoke threshold."""
        pop = Population(AstrocyteNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        assert neuron_steps / elapsed > 2000


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
