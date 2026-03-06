# SPDX-License-Identifier: AGPL-3.0-or-later
"""
NeuroBench-Aligned Benchmark Harness
======================================

Platform-agnostic metrics aligned with the NeuroBench methodology
(Yik et al., 2023 — arXiv:2304.04640).

Metrics computed:
  - Activation sparsity (fraction of silent neurons per timestep)
  - Synaptic operations (total MAC-equivalent ops)
  - Connection sparsity (fraction of zero weights)
  - Latency per timestep (wall-clock)
  - Energy proxy (ops × latency, dimensionless)
  - Memory footprint (weight matrix bytes)

Usage::

    python benchmarks/neurobench_harness.py
    python benchmarks/neurobench_harness.py --markdown
    python benchmarks/neurobench_harness.py --json results.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class NeuroBenchMetrics:
    model_name: str
    n_neurons: int
    n_synapses: int
    bitstream_length: int
    timesteps: int
    activation_sparsity: float
    connection_sparsity: float
    synaptic_ops_total: int
    latency_per_step_us: float
    energy_proxy: float
    memory_bytes: int
    wall_clock_s: float
    throughput_mops: float


def measure_sc_dense(
    n_inputs: int = 16,
    n_neurons: int = 8,
    length: int = 256,
    timesteps: int = 100,
    seed: int = 42,
) -> NeuroBenchMetrics:
    from sc_neurocore import SCDenseLayer

    rng = np.random.default_rng(seed)
    x_inputs = rng.uniform(0.1, 0.9, n_inputs).tolist()
    weight_values = rng.uniform(-0.5, 0.5, n_inputs).tolist()

    layer = SCDenseLayer(
        n_neurons=n_neurons,
        x_inputs=x_inputs,
        weight_values=weight_values,
        x_min=0.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        length=length,
    )

    # Connection sparsity: fraction of near-zero weights
    w_arr = np.array(weight_values)
    conn_sparsity = float(np.mean(np.abs(w_arr) < 1e-6))

    n_synapses = n_inputs * n_neurons
    # Each SC synapse = 1 AND gate per bit = length ops per synapse per step
    ops_per_step = n_synapses * length
    total_ops = ops_per_step * timesteps

    spike_counts = np.zeros(n_neurons, dtype=int)

    t0 = time.perf_counter()
    for _ in range(timesteps):
        layer_copy = SCDenseLayer(
            n_neurons=n_neurons,
            x_inputs=x_inputs,
            weight_values=weight_values,
            x_min=0.0,
            x_max=1.0,
            w_min=-1.0,
            w_max=1.0,
            length=length,
        )
        layer_copy.run(T=1)
        trains = layer_copy.get_spike_trains()
        for i, train in enumerate(trains):
            if len(train) > 0:
                spike_counts[i] += 1
    wall = time.perf_counter() - t0

    activation_sparsity = 1.0 - float(np.mean(spike_counts > 0))
    latency_us = (wall / timesteps) * 1e6
    energy_proxy = total_ops * (wall / timesteps)
    memory = n_synapses * 8  # f64 weights

    return NeuroBenchMetrics(
        model_name=f"SCDenseLayer({n_inputs}×{n_neurons}, L={length})",
        n_neurons=n_neurons,
        n_synapses=n_synapses,
        bitstream_length=length,
        timesteps=timesteps,
        activation_sparsity=activation_sparsity,
        connection_sparsity=conn_sparsity,
        synaptic_ops_total=total_ops,
        latency_per_step_us=latency_us,
        energy_proxy=energy_proxy,
        memory_bytes=memory,
        wall_clock_s=wall,
        throughput_mops=total_ops / wall / 1e6 if wall > 0 else 0.0,
    )


def measure_vectorized(
    n_inputs: int = 32,
    n_neurons: int = 16,
    length: int = 1024,
    timesteps: int = 50,
    seed: int = 42,
) -> NeuroBenchMetrics:
    from sc_neurocore import VectorizedSCLayer

    rng = np.random.default_rng(seed)
    layer = VectorizedSCLayer(n_inputs=n_inputs, n_neurons=n_neurons, length=length)
    inputs = rng.uniform(0.1, 0.9, n_inputs).tolist()

    n_synapses = n_inputs * n_neurons
    ops_per_step = n_synapses * length
    total_ops = ops_per_step * timesteps

    active_count = 0

    t0 = time.perf_counter()
    for _ in range(timesteps):
        out = layer.forward(inputs)
        active_count += int(np.sum(np.array(out) > 0))
    wall = time.perf_counter() - t0

    activation_sparsity = 1.0 - (active_count / (timesteps * n_neurons))
    latency_us = (wall / timesteps) * 1e6
    energy_proxy = total_ops * (wall / timesteps)
    memory = n_synapses * 8

    return NeuroBenchMetrics(
        model_name=f"VectorizedSCLayer({n_inputs}×{n_neurons}, L={length})",
        n_neurons=n_neurons,
        n_synapses=n_synapses,
        bitstream_length=length,
        timesteps=timesteps,
        activation_sparsity=activation_sparsity,
        connection_sparsity=0.0,
        synaptic_ops_total=total_ops,
        latency_per_step_us=latency_us,
        energy_proxy=energy_proxy,
        memory_bytes=memory,
        wall_clock_s=wall,
        throughput_mops=total_ops / wall / 1e6 if wall > 0 else 0.0,
    )


BENCH_SUITE = [
    ("sc_dense_small", lambda: measure_sc_dense(n_inputs=8, n_neurons=4, length=256, timesteps=50)),
    (
        "sc_dense_medium",
        lambda: measure_sc_dense(n_inputs=16, n_neurons=8, length=512, timesteps=30),
    ),
    (
        "vectorized_small",
        lambda: measure_vectorized(n_inputs=16, n_neurons=8, length=512, timesteps=50),
    ),
    (
        "vectorized_large",
        lambda: measure_vectorized(n_inputs=64, n_neurons=32, length=1024, timesteps=20),
    ),
]


def format_markdown(results: list[NeuroBenchMetrics]) -> str:
    lines = [
        "| Model | Neurons | SynOps | Act. Sparsity | Latency (µs) | Throughput (MOP/s) | Memory (B) |",
        "|-------|--------:|-------:|--------------:|-------------:|-------------------:|-----------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.n_neurons} | {r.synaptic_ops_total:,} "
            f"| {r.activation_sparsity:.2f} | {r.latency_per_step_us:.1f} "
            f"| {r.throughput_mops:.1f} | {r.memory_bytes:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="NeuroBench-aligned SC-NeuroCore benchmark")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json", type=str, help="write results to JSON file")
    args = ap.parse_args()

    print("NeuroBench-Aligned Benchmark Suite")
    print("=" * 40)

    results: list[NeuroBenchMetrics] = []
    for name, fn in BENCH_SUITE:
        print(f"  {name}...", end=" ", flush=True)
        r = fn()
        print(f"{r.latency_per_step_us:.1f} µs/step, {r.throughput_mops:.1f} MOP/s")
        results.append(r)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
