# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA Mapper Benchmark Suite

"""Benchmark: DNA mapper compilation, simulation, and validation performance.

Measures throughput across multiple circuit sizes, gate types, and
analysis engines. Outputs JSON results to benchmarks/results/.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    CrossHybridizationChecker,
    GF4ErrorCorrection,
    KineticSimulator,
    NoiseModel,
    estimate_cost,
    generate_protocol,
)


def _bench(fn, label: str, n_runs: int = 5) -> dict:
    """Run a function multiple times and return timing statistics."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return {
        "label": label,
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "n_runs": n_runs,
        "result": result,
    }


def bench_compilation():
    """Benchmark circuit compilation at various scales."""
    results = []
    for n_gates in [1, 5, 10, 25, 50]:
        gates = []
        prev = "IN"
        for i in range(n_gates):
            out = f"g{i}"
            if i % 3 == 0:
                gates.append({"type": "AND", "inputs": [prev, f"x{i}"], "output": out})
            elif i % 3 == 1:
                gates.append({"type": "OR", "inputs": [prev, f"x{i}"], "output": out})
            else:
                gates.append({"type": "NOT", "inputs": [prev], "output": out})
            prev = out

        inputs = ["IN"] + [f"x{i}" for i in range(n_gates) if i % 3 != 2]

        def compile_fn(g=gates, inp=inputs, p=prev):
            c = BitstreamToDNA(seed=42)
            return c.compile_network(g, inp, [p])

        r = _bench(compile_fn, f"compile_{n_gates}_gates")
        r["n_gates"] = n_gates
        design = r.pop("result")
        r["total_strands"] = design.total_strands
        r["total_nucleotides"] = design.total_nucleotides
        results.append(r)
        print(
            f"  compile {n_gates:3d} gates: {r['mean_s'] * 1000:.1f} ms "
            f"({r['total_strands']} strands, {r['total_nucleotides']} nt)"
        )

    return results


def bench_simulation():
    """Benchmark kinetic simulation at various durations."""
    c = BitstreamToDNA(seed=42)
    design = c.compile_network(
        gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
        input_names=["A", "B"],
        output_names=["C"],
    )
    sim = KineticSimulator()
    results = []

    for duration in [100, 1000, 3600, 7200]:

        def sim_fn(d=duration):
            return sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=float(d))

        r = _bench(sim_fn, f"simulate_{duration}s")
        r["duration_s"] = duration
        r.pop("result")
        results.append(r)
        print(f"  simulate {duration:5d}s: {r['mean_s'] * 1000:.1f} ms")

    return results


def bench_error_correction():
    """Benchmark GF(4) error correction encode/decode."""
    ec = GF4ErrorCorrection()
    results = []

    for seq_len in [48, 240, 1200]:
        seq = "ACGT" * (seq_len // 4)

        def encode_fn(s=seq):
            return ec.encode(s)

        r = _bench(encode_fn, f"ec_encode_{seq_len}nt", n_runs=20)
        r["seq_length"] = seq_len
        encoded = r.pop("result")
        r["encoded_length"] = len(encoded)
        results.append(r)

        def decode_fn(e=encoded):
            return ec.decode(e)

        r2 = _bench(decode_fn, f"ec_decode_{seq_len}nt", n_runs=20)
        r2["seq_length"] = seq_len
        r2.pop("result")
        results.append(r2)
        print(
            f"  EC {seq_len:5d} nt: encode {r['mean_s'] * 1e6:.0f} µs, "
            f"decode {r2['mean_s'] * 1e6:.0f} µs"
        )

    return results


def bench_cross_hybridization():
    """Benchmark cross-hybridization checking at scale."""
    results = []
    checker = CrossHybridizationChecker()

    for n_gates in [5, 10, 25]:
        c = BitstreamToDNA(seed=42)
        gates = []
        prev = "IN"
        for i in range(n_gates):
            out = f"g{i}"
            gates.append({"type": "NOT", "inputs": [prev], "output": out})
            prev = out
        design = c.compile_network(gates, ["IN"], [prev])

        def check_fn(d=design):
            return checker.check(d)

        r = _bench(check_fn, f"xhyb_{n_gates}_gates", n_runs=3)
        r["n_gates"] = n_gates
        flags = r.pop("result")
        r["n_flags"] = len(flags)
        results.append(r)
        print(
            f"  X-hyb {n_gates:3d} gates ({design.total_strands} strands): "
            f"{r['mean_s'] * 1000:.1f} ms, {r['n_flags']} flags"
        )

    return results


def bench_cost_and_protocol():
    """Benchmark cost estimation and protocol generation."""
    c = BitstreamToDNA(seed=42)
    design = c.compile_network(
        gates=[
            {"type": "AND", "inputs": ["A", "B"], "output": "X"},
            {"type": "OR", "inputs": ["X", "C"], "output": "Y"},
        ],
        input_names=["A", "B", "C"],
        output_names=["Y"],
    )

    r1 = _bench(lambda: estimate_cost(design), "cost_estimation", n_runs=50)
    cost = r1.pop("result")
    r1["total_cost_usd"] = cost["total_cost_usd"]
    print(f"  Cost estimation: {r1['mean_s'] * 1e6:.0f} µs (${cost['total_cost_usd']:.2f})")

    r2 = _bench(lambda: generate_protocol(design), "protocol_gen", n_runs=50)
    protocol = r2.pop("result")
    r2["protocol_lines"] = len(protocol.split("\n"))
    print(f"  Protocol gen:    {r2['mean_s'] * 1e6:.0f} µs ({r2['protocol_lines']} lines)")

    return [r1, r2]


def main():
    print("=" * 60)
    print("SC-NeuroCore DNA Mapper Benchmark")
    print("=" * 60)
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print()

    all_results = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "benchmarks": {},
    }

    print("[1/5] Compilation benchmarks")
    all_results["benchmarks"]["compilation"] = bench_compilation()

    print("\n[2/5] Simulation benchmarks")
    all_results["benchmarks"]["simulation"] = bench_simulation()

    print("\n[3/5] Error correction benchmarks")
    all_results["benchmarks"]["error_correction"] = bench_error_correction()

    print("\n[4/5] Cross-hybridization benchmarks")
    all_results["benchmarks"]["cross_hybridization"] = bench_cross_hybridization()

    print("\n[5/5] Cost & protocol benchmarks")
    all_results["benchmarks"]["cost_protocol"] = bench_cost_and_protocol()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dna_mapper_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults written to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
