#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-Framework SNN Benchmark

"""
Cross-Framework SNN Benchmark
==============================

Balanced E-I LIF network at multiple scales. Measures wall time, peak
memory, and spike rate across frameworks. Honest comparison including
Brian2's C++ standalone mode.

Frameworks tested:
  - SC-NeuroCore (NumPy backend)
  - SC-NeuroCore (Rust engine, if installed)
  - Brian2 (runtime mode — interpreted C++)
  - Brian2 (C++ standalone — compiled, fastest Brian2 path)
  - snnTorch (PyTorch CPU)
  - Norse (PyTorch CPU)

Network: Brunel-like balanced E-I LIF, 80/20 split, 10% random
connectivity, Poisson external drive, 300ms simulation, dt=0.1ms.

Usage::

    python benchmarks/cross_framework_benchmark.py
    python benchmarks/cross_framework_benchmark.py --scales 1000 5000
    python benchmarks/cross_framework_benchmark.py --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchResult:
    framework: str
    mode: str
    n_neurons: int
    wall_time_s: float
    peak_memory_mb: float
    n_spikes: int
    rate_hz: float
    error: str | None = None


def _measure(fn, label: str, n_neurons: int, mode: str) -> BenchResult:
    """Run fn(), measure time and memory."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        n_spikes, rate = fn()
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return BenchResult(
            framework=label,
            mode=mode,
            n_neurons=n_neurons,
            wall_time_s=round(dt, 4),
            peak_memory_mb=round(peak / 1e6, 1),
            n_spikes=n_spikes,
            rate_hz=round(rate, 2),
        )
    except Exception as e:
        tracemalloc.stop()
        return BenchResult(
            framework=label,
            mode=mode,
            n_neurons=n_neurons,
            wall_time_s=0,
            peak_memory_mb=0,
            n_spikes=0,
            rate_hz=0,
            error=str(e)[:200],
        )


# --- SC-NeuroCore NumPy ---


def bench_sc_numpy(n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1) -> BenchResult:
    def run():
        from sc_neurocore.studio.network import _simulate_numpy

        n_exc = int(n_neurons * 0.8)
        n_inh = n_neurons - n_exc
        r = _simulate_numpy(
            n_exc=n_exc,
            n_inh=n_inh,
            w_ee=0.1,
            w_ei=0.4,
            w_ie=0.1,
            w_ii=0.4,
            p_conn=0.1,
            ext_rate=8.0,
            duration=duration_ms,
            dt=dt,
        )
        n_spikes = r["n_spikes"]
        rate = n_spikes / (n_neurons * duration_ms / 1000)
        return n_spikes, rate

    return _measure(run, "SC-NeuroCore", "NumPy", n_neurons)


# --- SC-NeuroCore Rust ---


def bench_sc_rust(n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1) -> BenchResult:
    def run():
        from sc_neurocore.studio.network import _simulate_rust

        n_exc = int(n_neurons * 0.8)
        n_inh = n_neurons - n_exc
        r = _simulate_rust(
            n_exc=n_exc,
            n_inh=n_inh,
            w_ee=0.1,
            w_ei=0.4,
            w_ie=0.1,
            w_ii=0.4,
            p_conn=0.1,
            ext_rate=8.0,
            duration=duration_ms,
            dt=dt,
        )
        n_spikes = r["n_spikes"]
        rate = n_spikes / (n_neurons * duration_ms / 1000)
        return n_spikes, rate

    return _measure(run, "SC-NeuroCore", "Rust", n_neurons)


# --- Brian2 Runtime ---


def bench_brian2_runtime(
    n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1
) -> BenchResult:
    def run():
        import brian2

        brian2.prefs.codegen.target = "numpy"
        brian2.start_scope()

        n_exc = int(n_neurons * 0.8)
        n_inh = n_neurons - n_exc

        eqs = """
        dv/dt = (-v + v_rest) / tau_m : volt (unless refractory)
        """
        tau_m = 20 * brian2.ms
        v_rest = -65 * brian2.mV
        v_thresh = -50 * brian2.mV
        v_reset = -65 * brian2.mV

        G = brian2.NeuronGroup(
            n_neurons,
            eqs,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=2 * brian2.ms,
            method="euler",
        )
        G.v = v_rest

        exc = G[:n_exc]
        inh = G[n_exc:]

        w_e = 0.5 * brian2.mV
        w_i = -2.0 * brian2.mV

        S_ee = brian2.Synapses(exc, exc, on_pre="v += w_e")
        S_ee.connect(p=0.1)
        S_ei = brian2.Synapses(exc, inh, on_pre="v += w_e")
        S_ei.connect(p=0.1)
        S_ie = brian2.Synapses(inh, exc, on_pre="v += w_i")
        S_ie.connect(p=0.1)
        S_ii = brian2.Synapses(inh, inh, on_pre="v += w_i")
        S_ii.connect(p=0.1)

        P = brian2.PoissonInput(
            G, "v", N=int(n_neurons * 0.1), rate=8 * brian2.Hz, weight=0.5 * brian2.mV
        )

        M = brian2.SpikeMonitor(G)
        brian2.run(duration_ms * brian2.ms)

        n_spikes = M.num_spikes
        rate = n_spikes / (n_neurons * duration_ms / 1000)
        return n_spikes, rate

    return _measure(run, "Brian2", "runtime (NumPy)", n_neurons)


# --- Brian2 C++ Standalone ---


def bench_brian2_standalone(
    n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1
) -> BenchResult:
    def run():
        import brian2
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="brian2_standalone_")
        brian2.set_device("cpp_standalone", directory=tmpdir, build_on_run=True)
        brian2.start_scope()

        n_exc = int(n_neurons * 0.8)
        n_inh = n_neurons - n_exc

        eqs = """
        dv/dt = (-v + v_rest) / tau_m : volt (unless refractory)
        """
        tau_m = 20 * brian2.ms
        v_rest = -65 * brian2.mV
        v_thresh = -50 * brian2.mV
        v_reset = -65 * brian2.mV

        G = brian2.NeuronGroup(
            n_neurons,
            eqs,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=2 * brian2.ms,
            method="euler",
        )
        G.v = v_rest

        exc = G[:n_exc]
        inh = G[n_exc:]

        w_e = 0.5 * brian2.mV
        w_i = -2.0 * brian2.mV

        S_ee = brian2.Synapses(exc, exc, on_pre="v += w_e")
        S_ee.connect(p=0.1)
        S_ei = brian2.Synapses(exc, inh, on_pre="v += w_e")
        S_ei.connect(p=0.1)
        S_ie = brian2.Synapses(inh, exc, on_pre="v += w_i")
        S_ie.connect(p=0.1)
        S_ii = brian2.Synapses(inh, inh, on_pre="v += w_i")
        S_ii.connect(p=0.1)

        P = brian2.PoissonInput(
            G, "v", N=int(n_neurons * 0.1), rate=8 * brian2.Hz, weight=0.5 * brian2.mV
        )

        M = brian2.SpikeMonitor(G)
        brian2.run(duration_ms * brian2.ms)

        n_spikes = M.num_spikes
        rate = n_spikes / (n_neurons * duration_ms / 1000)

        brian2.device.reinit()
        brian2.device.activate()

        return n_spikes, rate

    return _measure(run, "Brian2", "C++ standalone", n_neurons)


# --- snnTorch ---


def bench_snntorch(n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1) -> BenchResult:
    def run():
        import torch
        import snntorch as snn

        n_steps = int(duration_ms / dt)
        n_exc = int(n_neurons * 0.8)
        n_inh = n_neurons - n_exc

        lif = snn.Leaky(beta=0.9, threshold=1.0)
        mem = torch.zeros(n_neurons)
        spk_count = 0

        W = torch.zeros(n_neurons, n_neurons)
        mask = torch.rand(n_neurons, n_neurons) < 0.1
        mask.fill_diagonal_(False)
        W[mask & (torch.arange(n_neurons).unsqueeze(0) < n_exc)] = 0.1
        W[mask & (torch.arange(n_neurons).unsqueeze(0) >= n_exc)] = -0.4

        for t in range(min(n_steps, 3000)):
            ext = torch.poisson(torch.full((n_neurons,), 8.0 * dt / 1000.0)) * 0.5
            syn = W @ (mem > 1.0).float()
            current = ext + syn
            spk, mem = lif(current, mem)
            spk_count += spk.sum().item()

        rate = spk_count / (n_neurons * min(n_steps, 3000) * dt / 1000)
        return int(spk_count), rate

    return _measure(run, "snnTorch", "PyTorch CPU", n_neurons)


# --- Norse ---


def bench_norse(n_neurons: int, duration_ms: float = 300.0, dt: float = 0.1) -> BenchResult:
    def run():
        import torch
        from norse.torch import LIFRecurrentCell, LIFParameters

        n_steps = int(duration_ms / dt)
        n_exc = int(n_neurons * 0.8)

        params = LIFParameters(tau_mem_inv=torch.tensor(1 / 0.02))
        cell = LIFRecurrentCell(n_neurons, n_neurons, p=params)
        state = None
        spk_count = 0

        for t in range(min(n_steps, 3000)):
            ext = torch.poisson(torch.full((1, n_neurons), 8.0 * dt / 1000.0)) * 0.5
            output, state = cell(ext, state)
            spk_count += output.sum().item()

        rate = spk_count / (n_neurons * min(n_steps, 3000) * dt / 1000)
        return int(spk_count), rate

    return _measure(run, "Norse", "PyTorch CPU", n_neurons)


def main():
    parser = argparse.ArgumentParser(description="Cross-framework SNN benchmark")
    parser.add_argument("--scales", nargs="+", type=int, default=[1000])
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--skip-standalone",
        action="store_true",
        help="Skip Brian2 C++ standalone (slow compilation)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cross-Framework SNN Benchmark")
    print(f"Platform: {platform.processor()} | Python {sys.version.split()[0]}")
    print(f"Scales: {args.scales} neurons | 300ms simulation | dt=0.1ms")
    print("=" * 70)

    results = []
    for n in args.scales:
        print(f"\n--- {n} neurons ---")

        benchmarks = [
            ("SC-NeuroCore NumPy", lambda n=n: bench_sc_numpy(n)),
            ("SC-NeuroCore Rust", lambda n=n: bench_sc_rust(n)),
            ("Brian2 runtime", lambda n=n: bench_brian2_runtime(n)),
            ("snnTorch", lambda n=n: bench_snntorch(n)),
            ("Norse", lambda n=n: bench_norse(n)),
        ]

        if not args.skip_standalone:
            benchmarks.insert(3, ("Brian2 C++ standalone", lambda n=n: bench_brian2_standalone(n)))

        for name, fn in benchmarks:
            print(f"  {name:30s} ... ", end="", flush=True)
            r = fn()
            if r.error:
                print(f"ERROR: {r.error[:60]}")
            else:
                print(
                    f"{r.wall_time_s:8.3f}s  {r.peak_memory_mb:7.1f}MB  {r.n_spikes:>8} spikes  {r.rate_hz:6.1f} Hz"
                )
            results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print(
        f"{'Framework':<20} {'Mode':<20} {'Neurons':>8} {'Time (s)':>10} {'Memory':>10} {'Spikes':>8}"
    )
    print("-" * 70)
    for r in results:
        if r.error:
            print(f"{r.framework:<20} {r.mode:<20} {r.n_neurons:>8} {'ERROR':>10}")
        else:
            print(
                f"{r.framework:<20} {r.mode:<20} {r.n_neurons:>8} {r.wall_time_s:>10.3f} {r.peak_memory_mb:>9.1f}M {r.n_spikes:>8}"
            )

    if args.json:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.processor(),
            "python": sys.version.split()[0],
            "results": [asdict(r) for r in results],
        }
        Path(args.json).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
