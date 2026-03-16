# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Head-to-head microbenchmark: SC-NeuroCore vs snnTorch.

Measures single-neuron latency, dense-layer throughput, and strong scaling.
Outputs JSON artifact to benchmarks/results/snntorch_vs_sc_microbench.json.

Usage:
    python benchmarks/snntorch_vs_sc_microbench.py [--runs 5] [--json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchRow:
    test_name: str
    framework: str
    n_inputs: int
    n_neurons: int
    steps: int
    wall_time_s: float
    wall_std_s: float
    total_spikes: int
    mean_rate: float
    time_per_step_us: float
    runs: list[float] = field(default_factory=list)


def _bench_sc_single_neuron(steps: int, n_runs: int) -> BenchRow:
    from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

    times, spikes_list = [], []
    for r in range(n_runs):
        neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, noise_std=0.02, seed=r)
        rng = np.random.default_rng(r)
        currents = 0.8 + 0.3 * rng.standard_normal(steps)
        t0 = time.perf_counter()
        total = sum(neuron.step(float(c)) for c in currents)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name="single_neuron",
        framework="sc_neurocore",
        n_inputs=1,
        n_neurons=1,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / steps, 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_snntorch_single_neuron(steps: int, n_runs: int) -> BenchRow:
    import torch
    import snntorch as snn

    times, spikes_list = [], []
    for r in range(n_runs):
        lif = snn.Leaky(beta=0.9, threshold=1.0)
        torch.manual_seed(r)
        currents = 0.8 + 0.3 * torch.randn(steps)
        mem = torch.zeros(1)
        t0 = time.perf_counter()
        total = 0
        for t in range(steps):
            spk, mem = lif(currents[t].unsqueeze(0), mem)
            total += spk.item()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name="single_neuron",
        framework="snntorch",
        n_inputs=1,
        n_neurons=1,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / steps, 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_sc_dense(n_in: int, n_out: int, steps: int, n_runs: int) -> BenchRow:
    from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

    times, spikes_list = [], []
    for r in range(n_runs):
        layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=512, use_gpu=False)
        rng = np.random.default_rng(r)
        inputs = rng.uniform(0.3, 0.7, (steps, n_in))
        t0 = time.perf_counter()
        total = 0.0
        for t in range(steps):
            out = layer.forward(inputs[t])
            total += out.sum()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name="dense_layer",
        framework="sc_neurocore",
        n_inputs=n_in,
        n_neurons=n_out,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_out), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_snntorch_dense(n_in: int, n_out: int, steps: int, n_runs: int) -> BenchRow:
    import torch
    import torch.nn as nn
    import snntorch as snn

    times, spikes_list = [], []
    for r in range(n_runs):
        torch.manual_seed(r)
        fc = nn.Linear(n_in, n_out)
        lif = snn.Leaky(beta=0.9, threshold=1.0)
        rng = np.random.default_rng(r)
        inputs = torch.tensor(rng.uniform(0.3, 0.7, (steps, n_in)), dtype=torch.float32)
        mem = torch.zeros(n_out)
        t0 = time.perf_counter()
        total = 0.0
        with torch.no_grad():
            for t in range(steps):
                cur = fc(inputs[t])
                spk, mem = lif(cur, mem)
                total += spk.sum().item()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name="dense_layer",
        framework="snntorch",
        n_inputs=n_in,
        n_neurons=n_out,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_out), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_sc_scaling(n_neurons: int, steps: int, n_runs: int) -> BenchRow:
    from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

    times, spikes_list = [], []
    for r in range(n_runs):
        layer = VectorizedSCLayer(
            n_inputs=n_neurons, n_neurons=n_neurons, length=256, use_gpu=False
        )
        rng = np.random.default_rng(r)
        inputs = rng.uniform(0.2, 0.5, (steps, n_neurons))
        t0 = time.perf_counter()
        total = 0.0
        for t in range(steps):
            out = layer.forward(inputs[t])
            total += out.sum()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name=f"scaling_{n_neurons}",
        framework="sc_neurocore",
        n_inputs=n_neurons,
        n_neurons=n_neurons,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_neurons), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_snntorch_scaling(n_neurons: int, steps: int, n_runs: int) -> BenchRow:
    import torch
    import torch.nn as nn
    import snntorch as snn

    times, spikes_list = [], []
    for r in range(n_runs):
        torch.manual_seed(r)
        fc = nn.Linear(n_neurons, n_neurons)
        lif = snn.Leaky(beta=0.9, threshold=1.0)
        rng = np.random.default_rng(r)
        inputs = torch.tensor(rng.uniform(0.2, 0.5, (steps, n_neurons)), dtype=torch.float32)
        mem = torch.zeros(n_neurons)
        t0 = time.perf_counter()
        total = 0.0
        with torch.no_grad():
            for t in range(steps):
                cur = fc(inputs[t])
                spk, mem = lif(cur, mem)
                total += spk.sum().item()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name=f"scaling_{n_neurons}",
        framework="snntorch",
        n_inputs=n_neurons,
        n_neurons=n_neurons,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_neurons), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _try_rust_engine():
    """Return True if Rust SIMD engine is available."""
    try:
        from sc_neurocore_engine import sc_neurocore_engine  # noqa: F401

        return True
    except ImportError:
        return False


def _bench_rust_dense(n_in: int, n_out: int, steps: int, n_runs: int) -> BenchRow:
    from sc_neurocore_engine.sc_neurocore_engine import DenseLayer

    times, spikes_list = [], []
    for r in range(n_runs):
        layer = DenseLayer(n_in, n_out, 512, r)
        rng = np.random.default_rng(r)
        inputs = rng.uniform(0.3, 0.7, (steps, n_in))
        t0 = time.perf_counter()
        total = 0.0
        for t in range(steps):
            out = layer.forward_numpy(inputs[t])
            total += out.sum()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name="dense_layer",
        framework="sc_rust_simd",
        n_inputs=n_in,
        n_neurons=n_out,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_out), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def _bench_rust_scaling(n_neurons: int, steps: int, n_runs: int) -> BenchRow:
    from sc_neurocore_engine.sc_neurocore_engine import DenseLayer

    times, spikes_list = [], []
    for r in range(n_runs):
        layer = DenseLayer(n_neurons, n_neurons, 256, r)
        rng = np.random.default_rng(r)
        inputs = rng.uniform(0.2, 0.5, (steps, n_neurons))
        t0 = time.perf_counter()
        total = 0.0
        for t in range(steps):
            out = layer.forward_numpy(inputs[t])
            total += out.sum()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        spikes_list.append(total)

    wall = np.mean(times)
    return BenchRow(
        test_name=f"scaling_{n_neurons}",
        framework="sc_rust_simd",
        n_inputs=n_neurons,
        n_neurons=n_neurons,
        steps=steps,
        wall_time_s=round(wall, 6),
        wall_std_s=round(float(np.std(times)), 6),
        total_spikes=int(np.mean(spikes_list)),
        mean_rate=round(float(np.mean(spikes_list)) / (steps * n_neurons), 4),
        time_per_step_us=round(wall / steps * 1e6, 2),
        runs=[round(t, 6) for t in times],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--scales", nargs="+", type=int, default=[100, 500, 1000])
    args = parser.parse_args()

    rows: list[BenchRow] = []
    N = args.runs
    has_rust = _try_rust_engine()

    print("=== SC-NeuroCore vs snnTorch Microbenchmark ===")
    print(f"    Rust SIMD engine: {'YES' if has_rust else 'NO'}\n")

    # Test 1: Single neuron
    print("[1/3] Single neuron (1000 steps)...")
    rows.append(_bench_sc_single_neuron(1000, N))
    rows.append(_bench_snntorch_single_neuron(1000, N))
    print(f"  SC-NeuroCore: {rows[-2].time_per_step_us:.1f} us/step")
    print(f"  snnTorch:     {rows[-1].time_per_step_us:.1f} us/step")

    # Test 2: Dense layer 100->50
    print("\n[2/3] Dense layer 100->50 (500 steps)...")
    rows.append(_bench_sc_dense(100, 50, 500, N))
    rows.append(_bench_snntorch_dense(100, 50, 500, N))
    print(f"  SC-NeuroCore: {rows[-2].time_per_step_us:.1f} us/step")
    print(f"  snnTorch:     {rows[-1].time_per_step_us:.1f} us/step")
    if has_rust:
        rows.append(_bench_rust_dense(100, 50, 500, N))
        print(f"  SC Rust SIMD: {rows[-1].time_per_step_us:.1f} us/step")

    # Test 3: Scaling
    for scale in args.scales:
        steps = max(50, 500 // (scale // 100))
        print(f"\n[3/3] Scaling {scale}->{scale} ({steps} steps)...")
        rows.append(_bench_sc_scaling(scale, steps, N))
        rows.append(_bench_snntorch_scaling(scale, steps, N))
        print(f"  SC-NeuroCore: {rows[-2].time_per_step_us:.1f} us/step")
        print(f"  snnTorch:     {rows[-1].time_per_step_us:.1f} us/step")
        if has_rust:
            rows.append(_bench_rust_scaling(scale, steps, N))
            print(f"  SC Rust SIMD: {rows[-1].time_per_step_us:.1f} us/step")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Test':<20} {'Framework':<15} {'us/step':>10} {'spikes':>10} {'rate':>8}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r.test_name:<20} {r.framework:<15} {r.time_per_step_us:>10.1f} {r.total_spikes:>10} {r.mean_rate:>8.3f}"
        )

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
        },
        "n_runs": N,
        "rows": [asdict(r) for r in rows],
    }

    out_path = Path("benchmarks/results/snntorch_vs_sc_microbench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
