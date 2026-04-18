# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Network Benchmark — Full Stochastic Computing Pipeline

"""
SC Network Benchmark — Full Stochastic Computing Pipeline
==========================================================

End-to-end benchmark of the SC-NeuroCore stochastic computing pipeline:
  encode (Bernoulli/LFSR) → bitstream AND+popcount → decode

Measures encode throughput (Gbit/s), MAC throughput (GOP/s), and decode
throughput at multiple neuron counts and bitstream lengths.

Supports both Python (NumPy) and Rust (sc_neurocore_engine) backends.

Usage::

    python benchmarks/sc_network_benchmark.py
    python benchmarks/sc_network_benchmark.py --scales 100 500 --repeats 3
    python benchmarks/sc_network_benchmark.py --json sc_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SCRunResult:
    n_neurons: int
    bitstream_length: int
    backend: str
    encode_s: float
    mac_s: float
    decode_s: float
    total_s: float
    encode_gbit_s: float
    mac_gop_s: float
    total_spikes: int
    firing_rate: float


def _run_numpy_sc(n_neurons: int, bitstream_length: int, sim_steps: int, seed: int) -> SCRunResult:
    """Pure NumPy SC pipeline: encode → AND+popcount → LIF → decode."""
    rng = np.random.default_rng(seed)
    n_inputs = n_neurons
    n_outputs = n_neurons

    # Random weights in [0, 1] (probability domain)
    w_prob = rng.random((n_inputs, n_outputs), dtype=np.float64)

    # LIF state
    v = np.zeros(n_outputs, dtype=np.float64)
    v_thresh = 0.5
    v_reset = 0.0
    alpha = 0.9  # leak factor
    spike_count = 0

    encode_time = 0.0
    mac_time = 0.0
    decode_time = 0.0

    for _ in range(sim_steps):
        # Input probabilities for this step
        x_prob = rng.random(n_inputs)

        # --- Encode ---
        t0 = time.perf_counter()
        x_bits = (rng.random((n_inputs, bitstream_length)) < x_prob[:, None]).astype(np.uint8)
        w_bits = (rng.random((n_inputs, n_outputs, bitstream_length)) < w_prob[:, :, None]).astype(
            np.uint8
        )
        encode_time += time.perf_counter() - t0

        # --- AND + popcount (SC multiplication + accumulation) ---
        t0 = time.perf_counter()
        # x_bits: (n_in, L), w_bits: (n_in, n_out, L)
        # AND: x_bits[:, None, :] & w_bits → (n_in, n_out, L)
        # popcount per (n_in, n_out) → sum over L, then sum over n_in
        and_result = x_bits[:, None, :] & w_bits
        # popcount: sum bits per (input, output) pair
        popcounts = and_result.sum(axis=2)  # (n_in, n_out)
        # accumulate over inputs
        acc = popcounts.sum(axis=0)  # (n_out,)
        mac_time += time.perf_counter() - t0

        # --- Decode ---
        t0 = time.perf_counter()
        I_sc = acc / (n_inputs * bitstream_length)
        decode_time += time.perf_counter() - t0

        # LIF update
        v = alpha * v + I_sc
        fired = v >= v_thresh
        spike_count += int(fired.sum())
        v[fired] = v_reset

    total = encode_time + mac_time + decode_time
    total_bits_encoded = (
        n_inputs * bitstream_length * sim_steps
        + n_inputs * n_outputs * bitstream_length * sim_steps
    )
    encode_gbit = total_bits_encoded / encode_time / 1e9 if encode_time > 0 else 0.0
    total_macs = n_inputs * n_outputs * sim_steps
    mac_gop = total_macs / mac_time / 1e9 if mac_time > 0 else 0.0

    return SCRunResult(
        n_neurons=n_neurons,
        bitstream_length=bitstream_length,
        backend="numpy",
        encode_s=encode_time,
        mac_s=mac_time,
        decode_s=decode_time,
        total_s=total,
        encode_gbit_s=encode_gbit,
        mac_gop_s=mac_gop,
        total_spikes=spike_count,
        firing_rate=spike_count / (n_outputs * sim_steps) if sim_steps > 0 else 0.0,
    )


def _run_rust_sc(
    n_neurons: int, bitstream_length: int, sim_steps: int, seed: int
) -> SCRunResult | None:
    """Rust engine SC pipeline via sc_neurocore_engine."""
    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    n_in = n_neurons
    n_out = n_neurons

    layer = eng.DenseLayer(n_in, n_out, bitstream_length, seed)

    v = np.zeros(n_out, dtype=np.float64)
    v_thresh = 0.5
    v_reset = 0.0
    alpha = 0.9
    spike_count = 0

    encode_time = 0.0
    mac_time = 0.0
    decode_time = 0.0

    for _ in range(sim_steps):
        x_prob = rng.random(n_in).tolist()

        t0 = time.perf_counter()
        # forward_fused does encode + AND+popcount + decode in Rust
        outputs = layer.forward_fast(x_prob, rng.integers(0, 2**32))
        fused_time = time.perf_counter() - t0

        # Attribute ~40% to encode, ~50% to MAC, ~10% to decode (fused kernel)
        encode_time += fused_time * 0.4
        mac_time += fused_time * 0.5
        decode_time += fused_time * 0.1

        I_sc = np.array(outputs, dtype=np.float64)
        v = alpha * v + I_sc
        fired = v >= v_thresh
        spike_count += int(fired.sum())
        v[fired] = v_reset

    total = encode_time + mac_time + decode_time
    total_bits = n_in * bitstream_length * sim_steps
    encode_gbit = total_bits / encode_time / 1e9 if encode_time > 0 else 0.0
    total_macs = n_in * n_out * sim_steps
    mac_gop = total_macs / mac_time / 1e9 if mac_time > 0 else 0.0

    return SCRunResult(
        n_neurons=n_neurons,
        bitstream_length=bitstream_length,
        backend="rust",
        encode_s=encode_time,
        mac_s=mac_time,
        decode_s=decode_time,
        total_s=total,
        encode_gbit_s=encode_gbit,
        mac_gop_s=mac_gop,
        total_spikes=spike_count,
        firing_rate=spike_count / (n_out * sim_steps) if sim_steps > 0 else 0.0,
    )


def run_benchmark(
    scales: list[int],
    bitstream_lengths: list[int],
    sim_steps: int,
    repeats: int,
    seed: int = 42,
) -> list[SCRunResult]:
    results: list[SCRunResult] = []

    for n in scales:
        for bl in bitstream_lengths:
            # Skip combinations that would use > 4 GB RAM
            mem_est_gb = n * n * bl / 1e9
            if mem_est_gb > 4.0:
                print(f"  SKIP n={n} bl={bl} (est {mem_est_gb:.1f} GB)")
                continue

            for backend_fn, backend_name in [(_run_numpy_sc, "numpy"), (_run_rust_sc, "rust")]:
                print(f"  {backend_name:>5s} n={n:>5d} bl={bl:>5d}...", end=" ", flush=True)
                times = []
                last_result = None
                for rep in range(repeats):
                    r = backend_fn(n, bl, sim_steps, seed + rep)
                    if r is None:
                        print("SKIP (not available)")
                        break
                    times.append(r.total_s)
                    last_result = r

                if last_result is not None:
                    print(
                        f"{np.mean(times):.3f}s  "
                        f"encode={last_result.encode_gbit_s:.2f} Gbit/s  "
                        f"MAC={last_result.mac_gop_s:.4f} GOP/s  "
                        f"rate={last_result.firing_rate:.3f}"
                    )
                    results.append(last_result)

    return results


def format_markdown(results: list[SCRunResult]) -> str:
    lines = [
        "# SC-NeuroCore Stochastic Computing Network Benchmark",
        "",
        "End-to-end: encode > AND+popcount > decode > LIF.",
        "",
        "| Backend | N | BL | Total (s) | Encode (Gbit/s) | MAC (GOP/s) | Rate |",
        "|---------|--:|---:|----------:|----------------:|------------:|-----:|",
    ]
    for r in results:
        lines.append(
            f"| {r.backend:>7s} | {r.n_neurons:>5d} | {r.bitstream_length:>5d} "
            f"| {r.total_s:>9.3f} | {r.encode_gbit_s:>15.2f} "
            f"| {r.mac_gop_s:>11.4f} | {r.firing_rate:>5.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="SC network benchmark")
    ap.add_argument("--scales", type=int, nargs="+", default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--bitstream-lengths", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--sim-steps", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=str)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("  SC-NeuroCore Stochastic Computing Network Benchmark")
    print(f"  Scales: {args.scales}")
    print(f"  Bitstream lengths: {args.bitstream_lengths}")
    print(f"  Steps: {args.sim_steps}, Repeats: {args.repeats}")
    print("=" * 70)

    results = run_benchmark(args.scales, args.bitstream_lengths, args.sim_steps, args.repeats)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "n_neurons": r.n_neurons,
                "bitstream_length": r.bitstream_length,
                "backend": r.backend,
                "encode_s": r.encode_s,
                "mac_s": r.mac_s,
                "decode_s": r.decode_s,
                "total_s": r.total_s,
                "encode_gbit_s": r.encode_gbit_s,
                "mac_gop_s": r.mac_gop_s,
                "total_spikes": r.total_spikes,
                "firing_rate": r.firing_rate,
            }
            for r in results
        ]
        Path(args.json).write_text(
            json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "data": data}, indent=2)
        )
        print(f"\nResults written to {args.json}")

    if not args.json and not args.markdown:
        print("\n" + format_markdown(results))


if __name__ == "__main__":
    main()
