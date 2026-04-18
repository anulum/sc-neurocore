# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench-compatible benchmark harness for SC-NeuroCore

"""NeuroBench-compatible benchmark harness for SC-NeuroCore.

Wraps SC-NeuroCore neuron models in the NeuroBench evaluation protocol
(connection count, MAC count, spike rate, latency, accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuroBenchMetrics:
    """Container for NeuroBench evaluation metrics."""

    connection_count: int = 0
    mac_count: int = 0
    spike_rate: float = 0.0
    latency_s: float = 0.0
    accuracy: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchCase:
    """Single benchmark case definition."""

    name: str
    n_neurons: int
    n_timesteps: int
    dataset: str = "smnist"


BENCH_SUITE: list[BenchCase] = [
    BenchCase("lif_256_25", 256, 25),
    BenchCase("lif_512_50", 512, 50),
    BenchCase("alif_256_25", 256, 25),
    BenchCase("conv_snn_28x28_25", 784, 25, dataset="mnist"),
]


def run_bench(case: BenchCase) -> NeuroBenchMetrics:
    """Run a single NeuroBench evaluation (stub — requires torch)."""
    return NeuroBenchMetrics(
        connection_count=case.n_neurons * case.n_neurons,
        mac_count=case.n_neurons * case.n_neurons * case.n_timesteps,
    )


if __name__ == "__main__":
    for c in BENCH_SUITE:
        m = run_bench(c)
        print(f"{c.name}: conns={m.connection_count}, MACs={m.mac_count}")
